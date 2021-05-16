#ifndef __HETU_ML_MODEL_MF_COMMON_SCHEDULER_H_
#define __HETU_ML_MODEL_MF_COMMON_SCHEDULER_H_

#include "common/logging.h"
#include "model/mf/common/common.h"
#include "model/mf/common/util.h"
#include <condition_variable>
#include <unordered_set>
#include <queue>
#include <mutex>

namespace hetu { 
namespace ml {
namespace mf {

class Scheduler
{
public:
  Scheduler(int nr_bins, int nr_threads, std::vector<int> cv_blocks)
  : nr_bins(nr_bins),
    nr_threads(nr_threads),
    nr_done_jobs(0),
    target(nr_bins*nr_bins),
    nr_paused_threads(0),
    terminated(false),
    counts(nr_bins*nr_bins, 0),
    busy_p_blocks(nr_bins, 0),
    busy_q_blocks(nr_bins, 0),
    block_losses(nr_bins*nr_bins, 0),
    block_errors(nr_bins*nr_bins, 0),
    cv_blocks(cv_blocks.begin(), cv_blocks.end()),
    distribution(0.0, 1.0) {
    for (int i = 0; i < nr_bins * nr_bins; ++i) {
      if (this->cv_blocks.find(i) == this->cv_blocks.end())
        pq.emplace(distribution(generator), i);
      block_generators.push_back(std::minstd_rand0(rand()));
    }
  }
  
  int get_job() {
    bool is_found = false;
    std::pair<float, int> block;

    while (!is_found) {
      std::lock_guard<std::mutex> lock(mtx);
      std::vector<std::pair<float, int>> locked_blocks;
      int p_block = 0;
      int q_block = 0;

      while (!pq.empty()) {
        block = pq.top();
        pq.pop();

        p_block = block.second / nr_bins;
        q_block = block.second % nr_bins;

        if (busy_p_blocks[p_block] || busy_q_blocks[q_block]) {
          locked_blocks.push_back(block);
        } else {
          busy_p_blocks[p_block] = 1;
          busy_q_blocks[q_block] = 1;
          counts[block.second] += 1;
          is_found = true;
          break;
        }
      }

      for (auto &block1 : locked_blocks)
        pq.push(block1);
    }

    return block.second;
  }
  
  int get_bpr_job(int first_block, bool is_column_oriented) {
    std::lock_guard<std::mutex> lock(mtx);
    int another = first_block;
    std::vector<std::pair<float, int>> locked_blocks;

    while (!pq.empty()) {
      std::pair<float, int> block = pq.top();
      pq.pop();

      int p_block = block.second/nr_bins;
      int q_block = block.second%nr_bins;

      auto is_rejected = [&] () {
        if(is_column_oriented)
          return first_block % nr_bins != q_block ||
            busy_p_blocks[p_block];
        else
          return first_block / nr_bins != p_block ||
            busy_q_blocks[q_block];
      };

      if (is_rejected()) {
        locked_blocks.push_back(block);
      } else {
        busy_p_blocks[p_block] = 1;
        busy_q_blocks[q_block] = 1;
        another = block.second;
        break;
      }
    }

    for (auto &block : locked_blocks)
      pq.push(block);

    return another;
  }
  
  void put_job(int block_idx, double loss, double error) {
    // Return the held block to the scheduler
    {
      std::lock_guard<std::mutex> lock(mtx);
      busy_p_blocks[block_idx / nr_bins] = 0;
      busy_q_blocks[block_idx % nr_bins] = 0;
      block_losses[block_idx] = loss;
      block_errors[block_idx] = error;
      ++nr_done_jobs;
      float priority = (float) counts[block_idx] + distribution(generator);
      pq.emplace(priority, block_idx);
      ++nr_paused_threads;
      // Tell others that a block is available again.
      cond_var.notify_all();
    }

    // Wait if nr_done_jobs (aka the number of processed blocks) is too many
    // because we want to print out the training status roughly once all blocks
    // are processed once. This is the only place that a solver thread should
    // wait for something.
    {
      std::unique_lock<std::mutex> lock(mtx);
      cond_var.wait(lock, [&] { return nr_done_jobs < target; });
    }

    // Nothing is blocking and this thread is going to take another block
    {
      std::lock_guard<std::mutex> lock(mtx);
      --nr_paused_threads;
    }
  }
  
  void put_bpr_job(int first_block, int second_block) {
    if (first_block == second_block)
      return;

    std::lock_guard<std::mutex> lock(mtx);
    {
      busy_p_blocks[second_block / nr_bins] = 0;
      busy_q_blocks[second_block % nr_bins] = 0;
      float priority = (float) counts[second_block] + distribution(generator);
      pq.emplace(priority, second_block);
    }
  }
  
  double get_loss() {
    std::lock_guard<std::mutex> lock(mtx);
    return std::accumulate(block_losses.begin(), block_losses.end(), 0.0);
  }
  
  double get_error() {
    std::lock_guard<std::mutex> lock(mtx);
    return std::accumulate(block_errors.begin(), block_errors.end(), 0.0);
  }
  
  int get_negative(int first_block, int second_block,
                   int m, int n, bool is_column_oriented) {
    int rand_val = (int) block_generators[first_block]();

    auto gen_random = [&] (int block_id) {
      int v_min, v_max;

      if (is_column_oriented) {
        int seg_size = (int) ceil((double) m / nr_bins);
        v_min = std::min((block_id / nr_bins) * seg_size, m - 1);
        v_max = std::min(v_min + seg_size, m - 1);
      } else {
        int seg_size = (int) ceil((double) n / nr_bins);
        v_min = std::min((block_id % nr_bins) * seg_size, n - 1);
        v_max = std::min(v_min + seg_size, n - 1);
      }
      if (v_max == v_min)
        return v_min;
      else
        return rand_val % (v_max - v_min) + v_min;
    };

    if (rand_val % 2)
      return (int) gen_random(first_block);
    else
      return (int) gen_random(second_block);
  }
  
  void wait_for_jobs_done() {
    std::unique_lock<std::mutex> lock(mtx);

    // The first thing the main thread should wait for is that solver threads
    // process enough matrix blocks.
    // [REVIEW] Is it really needed? Solver threads automatically stop if they
    // process too many blocks, so the next wait should be enough for stopping
    // the main thread when nr_done_job is not enough.
    cond_var.wait(lock, [&] { return nr_done_jobs >= target; });

    // Wait for all threads to stop. Once a thread realizes that all threads
    // have processed enough blocks it should stop. Then, the main thread can
    // print values safely.
    cond_var.wait(lock, [&] { return nr_paused_threads == nr_threads; });
  }
  
  void resume() {
    std::lock_guard<std::mutex> lock(mtx);
    target += nr_bins * nr_bins;
    cond_var.notify_all();
  }
  
  void terminate() {
    std::lock_guard<std::mutex> lock(mtx);
    terminated = true;
  }
  
  bool is_terminated() {
    std::lock_guard<std::mutex> lock(mtx);
    return terminated;
  }

private:
  int nr_bins;
  int nr_threads;
  int nr_done_jobs;
  int target;
  int nr_paused_threads;
  bool terminated;
  std::vector<int> counts;
  std::vector<int> busy_p_blocks;
  std::vector<int> busy_q_blocks;
  std::vector<double> block_losses;
  std::vector<double> block_errors;
  std::vector<std::minstd_rand0> block_generators;
  std::unordered_set<int> cv_blocks;
  std::mutex mtx;
  std::condition_variable cond_var;
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution;
  std::priority_queue<std::pair<float, int>,
                      std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>> pq;
};

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_SCHEDULER_H_
