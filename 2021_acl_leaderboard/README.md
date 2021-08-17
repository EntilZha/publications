# Compiling Paper

1. Install Poetry python-poetry.org
2. Install Anaconda python
3. Install yarn/nodejs https://yarnpkg.com/getting-started/install
4. Run `yarn install` in the paper repository root
5. Create environment: Run `conda create -n leaderboard python=3.8`
6. Activate environment: Run `conda activate leaderboard`
7. Clone the code repo somewhere: https://github.com/EntilZha/isicle
8. `cd` to the code repo
9. Install dependencies: Inside the code repo root, run `poetry install`
10. Download data (see instructions below)
11. `cd` to paper repo
12. Compile: in repository root, run `make 2021_acl_leaderboard.paper.pdf`

To compile after the first time, only steps 6 and 10 are necessary.

At the moment, some of the plots that depend on the main code repo have to be generated there. This primarily means running these commands in the code repo (after downloading data)

```bash
$ isicle plot
```

## Downloading data

To download the data necessary to create the figures and thus the paper, you will need to either download them (instructions below) or create a simlink from the `data/` directory in the code repo to `2021_acl_leaderboard/auto_data/data`

1. Install [Rclone](rclone.org/)
2. Ask Pedro for the correct S3 keys to access the bucket `umd-leaderboard`
3. Run `rclone configure`, name it `s3`, select `s3` compatible, enter the access key, secret key
4. Run `mkdir -p 2021_acl_leaderboard/auto_data/data`
5. Run `rclone copy s3:umd-leaderboard/data 2021_acl_leaderboard/auto_data/data`
