import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    @staticmethod
    def _landmarks_pos_training_candidate():
        return np.array([
            [ 0.10253578,  0.77317562],
            [-0.51139355, -0.73564299],
            [ 0.52492795,  0.48745992],
            [-0.36622628, -0.54898976],
            [ 0.11373174,  0.28591129],
            [-0.1189741 , -0.62058249],
            [ 0.78197624,  0.78693098],
            [-0.29010624, -0.53433687],
            [ 0.45388382,  0.12545045],
            [-0.55318211, -0.18298686],
        ])
    
    @staticmethod
    def _get_landmark_obs_color(k):
        return np.array([
            [0.22419692, 0.33154848, 0.94179808],
            [0.96827756, 0.58169505, 0.64084629],
            [0.96514044, 0.25477855, 0.60962737],
            [0.69233985, 0.79945287, 0.17091062],
            [0.07833477, 0.69906701, 0.71441469],
            [0.38310141, 0.06734659, 0.09906027],
            [0.67769081, 0.02953352, 0.30799422],
            [0.37452487, 0.14679729, 0.2327721 ],
            [0.5982687 , 0.29884011, 0.40461689],
            [0.44405978, 0.83097346, 0.66070286],
       ])[k]
    
    @staticmethod
    def _curriculum_on_landmark_size(difficulty):
        size_map = {
            "naive": 0.3,
            "easy": 0.2,
            "mid": 0.1,
            "hard": 0.05,
            "insane": 0.02,
        }
        return size_map[difficulty]
    
    def make_world(self, args, seed=None):
        world = World()
        world.args = args
        world.rng = np.random.RandomState(seed if seed else np.random.randint(1e9))
        world.world_length = args.episode_length
        world.collaborative = True
        # set any world properties first
        world.dim_c = args.max_landmarks if args.leader_signal else 1
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks
        
        assert args.num_agents <= args.max_agents, "num_agents exceeds limit"
        assert args.num_landmarks <= args.max_landmarks, "num_landmarks exceeds limit"
        assert args.max_landmarks <= 10, "too much ({} more than ten) landmarks".format(args.max_landmars)
        assert args.K_agents <= args.num_agents, "K_agents exceeds the number of agents"
        # add agents
        world.agents = [Agent() for _ in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = False
            agent.size = 0.05
            agent.color = np.zeros(3)
        
        # add landmarks
        world.landmarks = [Landmark() for _ in range(world.num_landmarks)]
        landmark_size = self._curriculum_on_landmark_size(args.landmark_size_curriculum)
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = landmark_size
        # make initial conditions
        self.reset_world(world)
        if args.fix_landmark:
            self._landmark_init(world) # if to fix the map in the future, make up the missing initialization first
        self._reset_target(world) # awalys do one at the very beginning
        return world
    
    def _landmark_init(self, world):
        if world.args.given_landmark_pos:
            pos_candi = self._landmarks_pos_training_candidate()
            assert world.args.num_landmarks < len(pos_candi), "landmarks more than candidate positions"
            chosen_pos = pos_candi[world.rng.choice(np.arange(pos_candi.shape[0]), world.args.num_landmarks, replace=False)]
        else:
            chosen_pos = 0.8 * world.rng.uniform(-1, +1, (world.args.num_landmarks, world.dim_p))
        for landmark, pos in zip(world.landmarks, chosen_pos):
            landmark.state.p_pos = np.copy(pos)
            landmark.state.p_vel = np.zeros(world.dim_p)
    
    @staticmethod
    def _reset_target(world):
        for k, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1., k / max(1, world.num_landmarks - 1), 0.25])
        target_id = world.args.given_target_id if world.args.given_target_id >= 0 else world.rng.randint(world.num_landmarks)
        world.landmarks[target_id].color = np.array([0.25, 0.25, 0.75])
    
    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.rng.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        if not world.args.fix_landmark:
            self._landmark_init(world)
        self._reset_target(world)

    @staticmethod
    def _dist(p0, p1):
        return ((p0.state.p_pos - p1.state.p_pos) ** 2).sum() ** .5
    
    @staticmethod
    def _is_target(landmark):
        return np.all(np.isclose(landmark.color, np.array([0.25, 0.25, 0.75])))
    
    def _find_target(self, world):
        for landmark in world.landmarks:
            if self._is_target(landmark):
                return landmark
    
    def _in_landmark(self, agent, landmark):
        return self._dist(agent, landmark) < landmark.size
    
    def _count_agent_in_landmark(self, landmark, world):
        cnt = 0
        for a in world.agents:
            if self._in_landmark(a, landmark):
                cnt += 1
        return cnt
    
    def reward(self, agent, world):
        """Agents are rewarded if
            exactly k agents are in the target
            max(0, 2 - dist_to_target) if there are less than k agents are in the target
        """
        target = self._find_target(world)
        cnt = self._count_agent_in_landmark(target, world)
        
        rew = 0
        if cnt == world.args.K_agents:
            rew += 1
        elif cnt > 0 and not world.args.extreme_sparse_reward:
            rew += 0.5
        if cnt < world.args.K_agents and world.args.distance_reward:
            rew += 0.2 * max(0, 1 - self._dist(agent, target))
        return rew

    def observation(self, agent, world):
        # get positions of all entities and agents in this agent's reference frame
        landmark_obs = []
        for k in range(world.args.max_landmarks):
            if k < world.args.num_landmarks:
                landmark_color = self._get_landmark_obs_color(k)
                if world.args.inform_target_color and self._is_target(world.landmarks[k]):
                    landmark_color = world.landmarks[k].color
                landmark_obs.append(np.concatenate([landmark_color, world.landmarks[k].state.p_pos - agent.state.p_pos]))
            else:
                landmark_obs.append(np.zeros(world.dim_p + 3))
        if world.args.permutate_landmarks:
            world.rng.shuffle(landmark_obs)    
                
        agent_obs = []
        for k in range(world.args.max_agents):
            if k < world.args.num_agents:
                if world.agents[k] is agent:
                    continue
                agent_obs.append(np.concatenate([np.ones(1), world.agents[k].state.p_pos - agent.state.p_pos]))
            else:
                agent_obs.append(np.zeros(world.dim_p + 1))
        if world.args.permutate_agents:
            world.rng.shuffle(agent_obs)        
        
        obs = landmark_obs + agent_obs
        
        # communication from the leader (agent[0])
        if world.args.leader_signal:
            obs.append(world.agents[0].state.c)
        
        return np.concatenate(obs)
    
    def info(self, agent, world):
        target = self._find_target(world)
        cnt = self._count_agent_in_landmark(target, world)
        reach_tar = 1. if self._in_landmark(agent, target) else 0.
        return {"reach_total": cnt, "reach_individual": reach_tar}
