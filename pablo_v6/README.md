# pablo_v6: feature, dead-ball and decoder code

Team code used by the pipeline in `../dfl/`: player and ball features (`scripts/explore/featlib.py`,
`fe_geometry.py`, `fb_ball.py`), goalkeeper clean-up (`gklib.py`), dead-ball features (`deadlib.py`), the joint H/A/D
decoder (`hmm3.py`, `scripts/make_submission.py`) and the ATD reader (`src/mtah_2027/`).

It is a snapshot of commit `524becb` of the team's working repository, reduced to the modules the pipeline imports,
with their code unchanged. Dependencies are pinned in `uv.lock` (`uv sync --frozen`).

Some of these modules also hold code from earlier experiments that the submission does not run: reading the public
ground truth (`common.gt_codes`, `featlib.labels`), referee-whistle detection from the video audio (`whistle_detect`,
`make_submission.build`), and a fixed second-half kick-off frame (`featlib.KICKOFF_2H`). The pipeline in `../dfl/`
replaces `featlib.labels` with a function that reads nothing. It sets the halves from the official graded blocks
(`games.half_video_blocks`) and uses only `make_submission.decode` and its constants. It also installs a hook that makes
any attempt to open a ground-truth file fail (`dfl/dfllib.py`, `forbid_public_truth`). See the [README at the root](../README.md) for setup and the command that
regenerates the submission.

`docs/more-than-a-hack-2027/` (a clone of the official repository) and `out/` are created during setup and are not
part of this repository.
