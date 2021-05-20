import functools
import os
import re

import pandas as pd


def time_to_seconds(tm_str):
    return sum(x * int(t) for x, t in zip([3600, 60, 1], tm_str.split(":")))


def main():
    """
    This script was made to convert the google sheets file containing the time steps for getup and sitdown to a format
    which is easier to process down the line.
    """
    df = pd.read_csv(os.path.expanduser("~/Downloads/tmp/gs_times.csv"))

    print(df.head())

    gu_pat = r"(Getup)(\d+)"
    sd_pat = r"(Sitdown)(\d+)"
    gu_reg = re.compile(gu_pat)
    sd_reg = re.compile(sd_pat)

    vid_dicts_list = []

    for idx, row in df.iterrows():
        all_time_points = []
        actions_dict = {"action": [], "act_idx": [], "time_step": [], "duration": []}
        for coln, val in row.items():
            if not pd.isnull(val):
                gu_match = gu_reg.match(coln)
                sd_match = sd_reg.match(coln)

                match = gu_match if gu_match is not None else (sd_match if sd_match is not None else None)
                if match is not None:
                    actions_dict["action"].append(match.group(1))
                    actions_dict["act_idx"].append(int(match.group(2)))
                    secs = time_to_seconds(val)
                    actions_dict["time_step"].append(secs)
                    all_time_points.append(secs)
        all_time_points.append(time_to_seconds(row["Duration"]))
        all_time_points = sorted(all_time_points)
        for tstep in actions_dict["time_step"]:
            next_ts = next(filter(lambda t: t > tstep, all_time_points))
            actions_dict["duration"].append(next_ts - tstep)
        actions_dict["vid_id"] = [row["VideoName"]] * len(actions_dict["action"])

        vid_dicts_list.append(actions_dict)

    vid_dict = {"vid_id":    functools.reduce(lambda l1, l2: l1 + l2, [dct["vid_id"] for dct in vid_dicts_list]),
                "action":    functools.reduce(lambda l1, l2: l1 + l2, [dct["action"] for dct in vid_dicts_list]),
                "act_idx":   functools.reduce(lambda l1, l2: l1 + l2, [dct["act_idx"] for dct in vid_dicts_list]),
                "time_step": functools.reduce(lambda l1, l2: l1 + l2, [dct["time_step"] for dct in vid_dicts_list]),
                "duration":  functools.reduce(lambda l1, l2: l1 + l2, [dct["duration"] for dct in vid_dicts_list])}

    out_df = pd.DataFrame(vid_dict)

    out_df.to_csv("getup_sitdown_table.csv")


if __name__ == "__main__":
    main()
