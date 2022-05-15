REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_offpolicy_runner import EpisodeRunner as OffPolicyRunner
REGISTRY["offpolicy"] = OffPolicyRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .maven_parallel_runner import ParallelRunner as MavenParallelRunner
from .maven_episode_runner import EpisodeRunner as MavenEpisodeRunner
REGISTRY["maven_parallel"] = MavenParallelRunner
REGISTRY["maven_episode"] = MavenEpisodeRunner
