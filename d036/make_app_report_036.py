# -*- coding: utf-8 -*-
# App copy of the 036 report: same text as d036/036_report.md, figure links replaced by artifact markers (art_<artifact id>,
# which track the latest version).  usage (workspace root): python d036/make_app_report_036.py
import re
ART = {"036_fig_data.png": "cba85741-a1df-4321-9dec-72aad073200c", "out_512/fig_compare_FOV_c1.png": "64567af7-fce6-4f49-a60b-44b174e1d8fd",
       "out_1024/fig_compare_FOV_c1.png": "70fd2fff-94df-460b-a6e3-e866b435d2eb", "036_fig_darkfield.png": "b74ae66b-56ed-44b3-8f47-ec6f978f6723",
       "036_fig_spokes.png": "0bf061cc-48db-4134-a2e0-4e6867e67361", "036_fig_pc_direct.png": "947f716a-ef33-42d8-8584-cf03d2d08a56",
       "036_fig_pc_compare.png": "77209410-6c25-4cbc-9bcb-ce933fdf4d1f", "036_fig_pc_starzoom.png": "80207f9f-9463-46f6-a0a2-3ce48b7b8ae7"}
s = open("d036/036_report.md", encoding="utf-8").read()
def sub(m):
    assert m.group(2) in ART, m.group(2)
    return f"![{m.group(1)}]({{{{artifact:art_{ART[m.group(2)]}}}}})"
s = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", sub, s)
MK = lambda aid: "{" * 2 + "artifact:art_" + aid + "}" * 2
s = s.replace("`out_512/fig_compare_planewave.png`", "[036_bf512_compare_planewave.png](" + MK("e3a2edee-d657-4cde-8a65-9477f3dba1bc") + ")")
s = s.replace("`out_1024/fig_compare_planewave.png`", "Drive の `figures/bf_1024_compare_planewave.png`")
open("art036/036_report.md", "w", encoding="utf-8").write(s); print("app copy written", s.count("{{artifact:"), "figure links")
