'''
Usage:
    Start a cluster on your cloud. Click Grafana. Copy and paste the grafana dashboard URL
    when using the script like below:

    python scrape.py
        --grafana-url="https://anyscale-internal-hsrczdm-0000-cust-mon.ray.aws-us-west-2-admin.anyscale-test-production.com/grafana/?orgId=1&token=wmmsuy03W0o-_GssZMMa4kRi3qY-310YnCPK-HBeBZs&var-ClusterID=ses_m9guYEViF2Th1UmN8XftgDrD&refresh=1m"
        --sess-csv=export.csv
        --output-dir=out

'''

import argparse
import csv
from datetime import datetime, timedelta
import json
import os
import requests
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

QUERY_PATH = "grafana/api/datasources/proxy/1/api/v1/query_range"
HTTP_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
}
# Cortex data is only available for the last 2 weeks.
DURATION = timedelta(days=14)


def query_url(url):
    r = urlparse(url)

    qs = parse_qs(r.query)
    assert "token" in qs, "URL must contains a token parameter."

    token_qs = urlencode({"token": qs["token"][0]})
    r = r._replace(path=QUERY_PATH)._replace(query=token_qs)

    return urlunparse(r)


def scrape_cluster(query_url, sess_id):
    QS = {
        "mem":
        f"sum(ray_node_mem_used{{relabelled_cluster_id=\"{sess_id}\", instance=~\".+\"}}) * 100 / "
        f"sum(ray_node_mem_total{{relabelled_cluster_id=\"{sess_id}\", instance=~\".+\"}})",
        "cpu": f"avg(ray_node_cpu_utilization{{relabelled_cluster_id=\"{sess_id}\", instance=~\".+\"}})",
        "gpu": f"avg(ray_node_gpus_utilization{{relabelled_cluster_id=\"{sess_id}\", instance=~\".+\"}})",
    }

    now = datetime.now()
    end = int(datetime.timestamp(now))
    start = int(datetime.timestamp(now - DURATION))

    res = {}
    for k, q in QS.items():
        data = urlencode({"query": q,
                          "start": start,
                          "end": end,
                          "step": "300"})
        resp = requests.post(query_url, data=data, headers=HTTP_HEADERS)
        res[k] = resp.json()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grafana-url", help="Copy your grafana URL and paste it here.")
    parser.add_argument(
        "--sess-csv", help="CSV file of cluster ids.")
    parser.add_argument(
        "--output-dir", help="Directory to write scraped JSON files.")
    args = parser.parse_args()

    cortex_url = query_url(args.grafana_url)

    with open(args.sess_csv, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            sess_id = row[0]
            d = scrape_cluster(cortex_url, sess_id)

            print(i, sess_id)
            with open(os.path.join(args.output_dir, sess_id + ",json"), "w") as of:
                of.write(json.dumps(d))
