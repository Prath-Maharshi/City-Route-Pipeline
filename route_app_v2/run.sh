#!/bin/bash
# Run v2 from the City_Route_Pipeline root directory
cd "$(dirname "$0")/.."
exec /home/mirmtech/anaconda3/bin/python3.10 route_app_v2/main.py "$@"
