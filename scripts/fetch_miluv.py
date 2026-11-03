#!/usr/bin/env python3
"""Fetch selected files (CSV/YAML) from MILUV experiment zips on figshare without downloading the images.

The zip is read through HTTP range requests (the figshare download URL redirects to a short-lived signed S3
URL, so every range request goes through the redirect again).

    python3 scripts/fetch_miluv.py --list default_1_random3_1
    python3 scripts/fetch_miluv.py default_1_random3_1 default_3_zigzag_1 --robots ifo001
"""
import argparse, io, json, os, re, sys, urllib.request, zipfile

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTICLE = 28386041


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size: return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:  # transient network errors
                if attempt == 4: raise
        n = len(data); b[:n] = data; self.pos += n
        return n


def file_index():
    url = f"https://api.figshare.com/v2/articles/{ARTICLE}/files?page_size=1000"
    with urllib.request.urlopen(url, timeout=60) as r:
        return {f["name"]: f for f in json.load(r)}


def open_zip(name, index):
    f = index[name + ".zip"]
    raw = HttpRangeFile(f"https://ndownloader.figshare.com/files/{f['id']}", f["size"])
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--list", action="store_true", help="list non-image members and exit")
    ap.add_argument("--robots", default="ifo001,ifo002,ifo003")
    ap.add_argument("--out", default=os.path.join(PKG, "data", "miluv"))
    a = ap.parse_args()
    index = file_index()
    robots = a.robots.split(",")
    for exp in a.experiments:
        z = open_zip(exp, index)
        members = [m for m in z.infolist() if not m.is_dir() and not re.search(r"\.(jpe?g|png|bag)$", m.filename, re.I)]
        if a.list:
            print(f"== {exp}: {len(z.infolist())} members, {len(members)} non-image")
            for m in members: print(f"   {m.filename:70s} {m.file_size/1e6:9.2f} MB")
            continue
        for m in members:
            parts = m.filename.split("/")
            rob = next((p for p in parts if p.startswith("ifo00")), None)
            if rob is not None and rob not in robots: continue
            if not re.search(r"\.(csv|yaml|yml|txt)$", m.filename, re.I): continue
            rel = m.filename[m.filename.index(exp):] if exp in m.filename else os.path.join(exp, m.filename)
            dst = os.path.join(a.out, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == m.file_size:
                print(f"   skip {rel}"); continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(m) as src, open(dst + ".part", "wb") as out:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk: break
                    out.write(chunk)
            os.replace(dst + ".part", dst)
            print(f"   got  {rel} ({m.file_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
