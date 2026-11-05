#pragma once

#include "session.h"

#include <map>
#include <string>

namespace kims {

class StreamSource {
public:
    virtual ~StreamSource() {}
    virtual bool pull(Session& S) = 0;
    virtual void onEstimate(double t_head, const double* p_head, const double* q_head,
                            double t_now, const double* p_now, const double* q_now) {}
};

int kimsRun(const std::map<std::string, std::string>& args, Session& S, StreamSource* src);

}
