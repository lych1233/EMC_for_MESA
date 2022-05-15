import numpy as np
from onpolicy.envs.explore.core import World, Agent, Landmark
from onpolicy.envs.explore.scenario import BaseScenario

import onpolicy.envs.explore.basic as basic


class Scenario(BaseScenario):
    def make_world(self, args, seed=None):
        world = World()
        world.args = args
        world.rng = np.random.RandomState(seed if seed else np.random.randint(1e9))
        world.world_length = args.episode_length
        world.collaborative = True
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks
        
        assert args.num_agents <= args.max_agents, "num_agents exceeds limit"
        assert args.num_landmarks <= args.max_landmarks, "num_landmarks exceeds limit"
        # assert args.K_agents <= args.num_agents, "K_agents exceeds the number of agents"
        
        # add agents
        world.agents = [Agent() for _ in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            agent.color = np.zeros(3)
        
        # add landmarks
        world.landmarks = [Landmark() for _ in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.25
            landmark.color = np.array([1., i / max(1, world.num_landmarks - 1), 0.25])
            landmark.state.p_pos = 0.8 * world.rng.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        target = world.landmarks[world.args.target_id]
        target.color = np.array([0.25, 0.25, 0.75])
        
        star = Landmark()
        star.name = 'star'
        star.collide = False
        star.movable = False
        star.size = 0.05
        star.color = np.array([0.25, 0.75, 0.25])
        star.state.p_pos = target.state.p_pos + target.size * world.rng.uniform(-1, +1, world.dim_p) / np.sqrt(2)
        star.state.p_vel = np.zeros(world.dim_p)
        world.landmarks.append(star)
        
        # make initial conditions
        self.reset_world(world)
        """if args.fix_landmark:
            self._landmark_init(world) # if to fix the map in the future, make up the missing initialization first
        self._reset_target(world) # awalys do one at the very beginning
        """
        return world

    @staticmethod
    def _dist(p0, p1):
        return ((p0.state.p_pos - p1.state.p_pos) ** 2).sum() ** .5
    
    def _pull_in(self, p, cir, dist_limit):
        d = self._dist(p, cir)
        if d > dist_limit:
            p.state.p_pos = cir.state.p_pos + (p.state.p_pos - cir.state.p_pos) / (d + 1e-8) * dist_limit
    
    """@staticmethod
    def _reset_target(world):
        for k, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1., k / max(1, world.num_landmarks - 1), 0.25])
        target_id = world.args.given_target_id if world.args.given_target_id >= 0 else world.rng.randint(world.num_landmarks)
        world.landmarks[target_id].color = np.array([0.25, 0.25, 0.75])"""
    
    def reset_world(self, world):
        # set random initial states
        world.success = 0
        """goal = world.landmarks[-1]
        goal.state.p_pos = 0.9 * world.rng.uniform(-1, +1, world.dim_p)
        goal.color = np.array([0.75, 0.25, 0.75])"""
        for agent in world.agents:
            agent.state.p_pos = world.rng.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            #agent.state.c = np.zeros(world.dim_c)
        """if not world.args.fix_landmark:
            self._landmark_init(world)
        self._reset_target(world)"""
        
        restrict_dis = 2 * np.sqrt(2) * world.args.difficulty
        # self._pull_in(goal, world.landmarks[world.args.target_id], restrict_dis)
        
        for agent in world.agents:
            agent.state.p_pos = world.rng.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            k = world.rng.randint(world.num_landmarks)
            cir_landmark = world.landmarks[-1] if k == world.args.target_id else world.landmarks[k]
            self._pull_in(agent, cir_landmark, restrict_dis)
    
    @staticmethod
    def _is_target(landmark):
        return np.all(np.isclose(landmark.color, np.array([0.25, 0.25, 0.75])))
    
    def _find_target(self, world):
        for landmark in world.landmarks[:world.num_landmarks]:
            if self._is_target(landmark):
                assert landmark is world.landmarks[world.args.target_id]
                return landmark
    
    def _in_landmark(self, agent, landmark):
        return self._dist(agent, landmark) < landmark.size
    
    def _count_agent_in_landmark(self, landmark, world):
        cnt = 0
        for a in world.agents:
            if self._in_landmark(a, landmark):
                cnt += 1
        return cnt
    
    """@staticmethod
    def _light_up(goal):
        return np.all(np.isclose(goal.color, np.array([0.25, 0.75, 0.25])))"""
    
    def reward(self, agent, world):
        """Agents are rewarded if
            exactly k agents are in the target
            max(0, 2 - dist_to_target) if there are less than k agents are in the target
        """
        rew = 0
        #goal = world.landmarks[-1]
        star = world.landmarks[-1]
        if self._count_agent_in_landmark(star, world) == world.num_agents:
            rew = 1
        """if cnt_in_target args.K_agents:
            rew = 1.
        elif cnt_in_target > 0:
            rew = 0
        else:
            for i in range(world.num_landmarks):
                landmark = world.landmarks[i]
                if landmark is target:
                    continue
                if self._count_agent_in_landmark(landmark, world) > 0:
                    rew = 0.5
                    break"""
        if world.success == world.num_agents:
            if world.args.spark_reward:
                rew = 0
        elif rew > 0.99:
            world.success += 1
        return rew

    def observation(self, agent, world):
        # check light up
        # goal = world.landmarks[-1]
        target = world.landmarks[world.args.target_id]
        # if self._count_agent_in_landmark(target, world) == world.num_agents:
        #     goal.color = np.array([0.25, 0.75, 0.25])
    
        # get positions of all entities and agents in this agent's reference frame
        landmark_obs = []
        for k in range(world.args.max_landmarks):
            if k < world.args.num_landmarks:
                landmark_color = basic.kth_rgb_color(k)
                if world.args.inform_target_color and world.landmarks[k] is target:
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
        
        """if self._light_up(goal):
            goal_obs = np.concatenate([goal.color, goal.state.p_pos - agent.state.p_pos])
        else:
            goal_obs = np.zeros(world.dim_p + 3)"""
        
        obs = landmark_obs + agent_obs # + [goal_obs]
        
        return np.concatenate(obs)
    
    def info(self, agent, world):
        target = self._find_target(world)
        cnt = self._count_agent_in_landmark(target, world)
        reach_tar = 1. if self._in_landmark(agent, target) else 0.
        return {"reach_total": cnt, "reach_individual": reach_tar, "success": world.success == world.num_agents}
