#ifndef __HETU_ML_COMMON_LOGGING_H_
#define __HETU_ML_COMMON_LOGGING_H_

#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <execinfo.h>

namespace hetu { 
namespace ml {

/******************************************************
 * Timing Utils
 ******************************************************/

#define COST_MSEC(_X)                                    \
  std::chrono::duration_cast<std::chrono::milliseconds>( \
    (_X##_stop) - (_X##_start)).count()
#define COST_SEC(_X)                                \
  std::chrono::duration_cast<std::chrono::seconds>( \
    (_X##_stop) - (_X##_start)).count()
#define TIK(_X)                                       \
  auto _X##_start = std::chrono::steady_clock::now(), \
    _X##_stop = _X##_start
#define TOK(_X) _X##_stop = std::chrono::steady_clock::now()


/******************************************************
 * Logging Utils
 ******************************************************/

#define __FILENAME__ (strrchr(__FILE__, '/')  \
  ? strrchr(__FILE__, '/') + 1                \
  : __FILE__)

#ifdef FL_NDEBUG
#define __LOG_TRACE \
  hetu::ml::MsgLogger(__FILENAME__, __LINE__, "TRACE", std::cout)
#define __LOG_DEBUG \
  hetu::ml::MsgLogger(__FILENAME__, __LINE__, "DEBUG", std::cout)
#else
#define __LOG_TRACE \
  hetu::ml::MsgLoggerVoidify(__FILENAME__, __LINE__, "TRACE")
#define __LOG_DEBUG \
  hetu::ml::MsgLoggerVoidify(__FILENAME__, __LINE__, "DEBUG")
#endif
#define __LOG_INFO \
  hetu::ml::MsgLogger(__FILENAME__, __LINE__, "INFO")
#define __LOG_WARN \
  hetu::ml::MsgLogger(__FILENAME__, __LINE__, "WARN")
#define __LOG_ERROR \
  hetu::ml::MsgLogger(__FILENAME__, __LINE__, "ERROR", std::cerr)
#define __LOG_FATAL \
  hetu::ml::FatalLogger(__FILENAME__, __LINE__, "FATAL", std::cerr)

#define HML_LOG_PREFIX "*HETU* "
#define HML_LOG(severity) __LOG_##severity.stream() << HML_LOG_PREFIX
#define HML_LOG_TRACE HML_LOG(TRACE)
#define HML_LOG_DEBUG HML_LOG(DEBUG)
#define HML_LOG_INFO HML_LOG(INFO)
#define HML_LOG_WARN HML_LOG(WARN)
#define HML_LOG_ERROR HML_LOG(ERROR)
#define HML_LOG_FATAL HML_LOG(FATAL)

inline void InitLogging(const char* arg) {
  // Pass
}

class MsgLogger {
public:
  MsgLogger(const char* file, int line, const char* level, 
            std::ostream& os = std::cout): _os(os) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = (std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000).count();
    this->_ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %X")
      << ":" << ms << "] [" << level << "] ";
      // << ":" << ms << " (" << file << ":" << line << ")] [" << level << "] ";
  }
  ~MsgLogger() { 
    this->_ss << "\n"; 
    this->_os << this->_ss.str();
  }
  inline std::ostringstream& stream() { return this->_ss; }
protected:
  std::ostringstream _ss;
  std::ostream& _os;
private:
  MsgLogger(const MsgLogger&);
  void operator=(const MsgLogger&);
};

inline std::string StackTrace() {
  std::ostringstream stacktrace_os;
  void *stack[10];
  int nframes = backtrace(stack, 10);
  stacktrace_os << "Stack trace returned " << nframes << " entries:" << "\n";
  char **msgs = backtrace_symbols(stack, nframes);
  if (msgs != nullptr) {
    for (int frameno = 0; frameno < nframes; ++frameno) {
      stacktrace_os << "[trace] (" << frameno << ") " << msgs[frameno] << "\n";
    }
  }
  free(msgs);
  std::string stack_trace = stacktrace_os.str();
  return stack_trace;
}

class FatalLogger {
public:
  FatalLogger(const char* file, int line, const char* level, 
              std::ostream& os = std::cerr): _os(os) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = (std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000).count();
    this->_ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %X")
      << ":" << ms << "] [" << level << "] ";
      // << ":" << ms << " (" << file << ":" << line << ")] [" << level << "] ";
  }
  ~FatalLogger() noexcept(false) { 
    this->_ss << "\n\n" << StackTrace() << "\n\n"; 
    this->_os << this->_ss.str();
    throw std::runtime_error(this->_ss.str());
  }
  inline std::ostringstream& stream() { return this->_ss; }
protected:
  std::ostringstream _ss;
  std::ostream& _os;
private:
  FatalLogger(const FatalLogger&);
  void operator=(const FatalLogger&);
};

class MsgLoggerVoidify {  // Empty logger
public:
  MsgLoggerVoidify(const char* file, int line, const char* level) {}
  ~MsgLoggerVoidify() {}
  inline std::ostringstream& stream() { return this->_ss; }
protected:
  std::ostringstream _ss;
private:
  MsgLoggerVoidify(const MsgLoggerVoidify&);
  void operator=(const MsgLoggerVoidify&);
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  if (vec.size() > 0) {
    os << vec[0];
    for (int i = 1; i < vec.size(); i++) 
      os << "," << vec[i];
  }
  os << "]";
  return os;
}

/******************************************************
 * Assertion Utils
 ******************************************************/

#define ASSERT(x) if (!(x)) HML_LOG_FATAL << "Assertion failed: "
#define ASSERT_LT(x, y) ASSERT((x) < (y))
#define ASSERT_GT(x, y) ASSERT((x) > (y))
#define ASSERT_LE(x, y) ASSERT((x) <= (y))
#define ASSERT_GE(x, y) ASSERT((x) >= (y))
#define ASSERT_EQ(x, y) ASSERT((x) == (y))
#define ASSERT_NE(x, y) ASSERT((x) != (y))
#define ASSERT_FUZZY_EQ(x, y, tol) ASSERT(std::abs((x) - (y)) < (tol))
#define ASSERT_FUZZY_NE(x, y, tol) ASSERT(std::abs((x) - (y)) >= (tol))

} // namespace ml
} // namespace hetu

#endif // __HETU_ML_COMMON_LOGGING_H_
