"""Download only matching output files of a Kaggle job, walking every page of the output listing.
Usage: python kaggle/pull.py KERNEL OUTDIR REGEX"""
import os
import re
import sys

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

kernel, out, pat = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
api = KaggleApi()
api.authenticate()
owner, slug = kernel.split("/")
token, n, seen = None, 0, 0
with api.build_kaggle_client() as k:
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = owner, slug
        if token:
            req.page_token = token
        for attempt in range(8):
            try:
                r = k.kernels.kernels_api_client.list_kernel_session_output(req)
                break
            except Exception as e:                       # noqa: BLE001 - flaky listing; retry with a pause
                import time
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"listing failed: {e}")
        for f in r.files or []:
            seen += 1
            if pat.search(f.file_name):
                p = os.path.join(out, f.file_name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                for attempt in range(8):
                    try:
                        data = requests.get(f.url, timeout=120).content
                        break
                    except Exception:                    # noqa: BLE001
                        import time
                        time.sleep(5 * (attempt + 1))
                with open(p, "wb") as fh:
                    fh.write(data)
                n += 1
        if r.log and not os.path.exists(os.path.join(out, slug + ".log")):
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, slug + ".log"), "w").write(r.log)
        token = r.next_page_token
        if not token:
            break
print(f"{kernel}: {n} files downloaded of {seen} listed")
