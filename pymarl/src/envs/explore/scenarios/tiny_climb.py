import numpy as np
from envs.explore.core import World, Agent, Landmark
from envs.explore.scenario import BaseScenario

import envs.explore.basic as basic


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.args = args
        world.rng = np.random.RandomState(args.seed)
        world.world_length = args.episode_length
        world.collaborative = True
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks
        world.game_states = {}
        
        assert world.collaborative, "climb game is fully cooperative"
        assert world.dim_p == 2, "only support generate 2D landmark posistions"

        assert args.num_agents <= args.max_agents, "num_agents exceeds limit"
        assert args.num_landmarks <= args.max_landmarks, "num_landmarks exceeds limit"
        assert args.K_agents <= args.num_agents, "K_agents exceeds the number of agents"
        assert not (args.spark_reward and args.mode_reward), "could only use either spark_reward or mode_reward"
        assert world.num_landmarks ** world.num_agents <= 10000, "too much (>10000) modes maybe!"
        
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
        self._create_map(world)
        world.landmarks[world.args.target_id].color = basic.render_target_color()
        self.reset_world(world)
        return world

    @staticmethod
    def _create_map(world):
        radius = 0.1
        positions = basic.initialize_non_overlapping_landmarks(world.num_landmarks, radius, rng=world.rng, difficulty=world.args.difficulty)
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = radius
            landmark.color = basic.render_kth_rgb_color(i)
            landmark.state.p_pos = positions[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.landmarks[world.args.target_id].color = basic.render_target_color()

    @staticmethod
    def _dist(p0, p1):
        return ((p0.state.p_pos - p1.state.p_pos) ** 2).sum() ** .5
    
    def _pull_in(self, p, cir, dist_limit):
        d = self._dist(p, cir)
        if d > dist_limit:
            p.state.p_pos = cir.state.p_pos + (p.state.p_pos - cir.state.p_pos) / (d + 1e-8) * dist_limit
    
    def reset_world(self, world):
        world.game_states["success"] = False
        world.game_states["mode_visits"] = np.zeros((world.num_landmarks ** world.num_agents), dtype=bool)
        world.game_states["current_mode"] = -1
        if world.args.reset_map:
            self._create_map(world)
        if world.args.reset_target_id:
            world.landmarks[world.args.target_id].color = basic.render_kth_rgb_color(world.args.target_id)
            world.args.target_id = world.rng.randint(world.args.num_agents)
            world.landmarks[world.args.target_id].color = basic.render_target_color()
        for agent in world.agents:
            agent.state.p_pos = world.rng.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        
        restrict_dis = 2 * np.sqrt(2) * world.args.difficulty        
        for agent in world.agents:
            agent.state.p_pos = world.rng.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            cir_landmark = world.landmarks[world.rng.randint(world.num_landmarks)]
            if world.args.difficulty < 0.999:
                self._pull_in(agent, cir_landmark, restrict_dis)
    
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
            all agents are in some landmark
            1.0: exactly k agents are in the target landmark
            0.5: none of the agents are in the target landmark
        """
        if world.agents[0] is not agent:
            return 0.0
        rew, cnt_in_target, mode_code = 0, 0, 0
        for a in world.agents:
            k = -1
            for i, landmark in enumerate(world.landmarks):
                if self._in_landmark(a, landmark):
                    k = i
                    break
            if k == -1:
                mode_code = -1
                break
            mode_code = mode_code * world.num_landmarks + k
            if k == world.args.target_id:
                cnt_in_target += 1
        world.game_states["current_mode"] = mode_code
        if mode_code == -1:
            rew = 0.0
            return rew
        
        if cnt_in_target == world.args.K_agents:
            rew = 1.0
        elif cnt_in_target == 0:
            rew = 0.5
        
        if world.args.spark_reward and (world.game_states["success"] or rew < 0.99):
            rew = 0.0
        if world.args.mode_reward and world.game_states["mode_visits"][mode_code]:
            rew = 0.0
        if rew > 0.99:
            world.game_states["success"] = True
        world.game_states["mode_visits"][mode_code] = True
        return rew

    def observation(self, agent, world):
        landmark_obs = []
        for i in range(world.args.max_landmarks):
            if i < world.args.num_landmarks:
                # landmark_color = basic.obs_kth_rgb_color(i)
                # if world.args.uncover_target_color and i == world.args.target_id:
                #     landmark_color = basic.obs_target_color()
                landmark_obs.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
                # landmark_obs.append(np.concatenate([landmark_color, world.landmarks[i].state.p_pos - agent.state.p_pos]))
            else:
                pass
                # landmark_obs.append(np.zeros(world.dim_p + 3))
        if world.args.permutate_landmarks:
            world.rng.shuffle(landmark_obs)    
                
        agent_obs = []
        for i in range(world.args.max_agents):
            if i < world.args.num_agents:
                if world.agents[i] is agent:
                    continue
                agent_obs.append(world.agents[i].state.p_pos - agent.state.p_pos)
                # agent_obs.append(np.concatenate([np.ones(1), world.agents[i].state.p_pos - agent.state.p_pos]))
            else:
                pass
                # agent_obs.append(np.zeros(world.dim_p + 1))
        if world.args.permutate_agents:
            world.rng.shuffle(agent_obs)
        
        obs = landmark_obs + agent_obs
        return np.concatenate(obs)
    
    def info(self, agent, world):
        if not world.args.elaborated_information:
            return {"mode_code": world.game_states["current_mode"]}
        target = world.landmarks[world.args.target_id]
        agents_in_target = self._count_agent_in_landmark(target, world)
        agents_in_landmarks = []
        for landmark in world.landmarks:
            agents_in_landmarks.append(self._count_agent_in_landmark(landmark, world))
        agents_in_landmark = np.array(agents_in_landmarks)
        reach_target = self._in_landmark(agent, target)
        return {
            "reach_target": reach_target,
            "agents_in_target": agents_in_target,
            "agents_in_landmark": agents_in_landmark,
            "goal_success": world.game_states["success"],
            "mode_code": world.game_states["current_mode"],
            "mode_success": world.game_states["mode_visits"],
        }
