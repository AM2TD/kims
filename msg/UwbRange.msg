# One UWB range measurement between a tag and an anchor.
# Driver-agnostic: any radio can publish this and kims_online will consume it.
Header header
float64 stamp       # sensor time [s], exact
int32   from_id     # tag id
int32   to_id       # anchor id
float64 range       # metres
float64 std         # measurement standard deviation, metres
