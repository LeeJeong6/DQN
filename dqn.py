import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import random
from collections import deque 

class GridWorld:
   
    actions = np.array([-1, 1])  
    states = np.array([0, 1, 2, 3, 4])  
    reward_map = np.array([2, 0, -1, -2 ,10])  
    
    def __init__(self):
        self.gamma = 0.9 
        self.value_map = np.zeros(5)  

    def get_next_state(self, state, action):
       
        next_state = state + action
        if next_state < 0 or next_state > 4:
            return state, -2  # 경계 벗어남: 상태 유지, 패널티 -2
        return next_state, -1  # 정상 이동: 패널티 없음

    def update_value(self):
        
        new_value_map = np.zeros_like(self.value_map)
        for state in self.states:
            value = 0
            for action in self.actions:
                next_state, penalty = self.get_next_state(state, action)
                reward = self.reward_map[next_state] + penalty
                value += 0.5 * (reward + self.gamma * self.value_map[next_state])
            new_value_map[state] = value
        self.value_map = new_value_map

    def simulate(self, iterations=100):

        for _ in range(iterations):
            self.update_value()
        return self.value_map

class TemporalDifference(GridWorld):
    def __init__(self):
        super().__init__()
        self.Q_map = np.zeros((5, 2))  # 상태(4) x 행동(2) 크기의 Q-테이블
        self.terminal_state = 4  # 종료 상태 정의

    def choose_action(self, state, epsilon=0.1):
        """ε-탐욕 정책으로 행동 선택 (인덱스 반환)"""
        if np.random.random() < epsilon:  # 탐험
            return np.random.randint(0, 2)  # 0 또는 1 반환
        else:  # 활용
            return np.argmax(self.Q_map[state])  # 최적 행동의 인덱스 반환

    def q_learning(self, alpha=0.1, num_episodes=1000, epsilon=0.2, start_state=1):
        """Q-learning 알고리즘 (에피소드 기반)"""
        for _ in range(num_episodes):
            state = start_state  # 에피소드 시작 상태
            while state != self.terminal_state:  
                # 1. 현재 상태에서 행동 선택
                action_idx = self.choose_action(state, epsilon)
                action = self.actions[action_idx]

                # 2. 다음 상태와 보상 계산
                next_state, penalty = self.get_next_state(state, action)
                reward = self.reward_map[next_state] + penalty

                # 3. 다음 상태가 종료 상태인지 확인하고 Q-값 계산
                if next_state == self.terminal_state:
                    max_next_q = 0  # 종료 상태에서는 미래 보상이 없음
                else:
                    max_next_q = np.max(self.Q_map[next_state])  # max_a' Q(s', a')

                # 4. Q-값 업데이트
                td_target = reward + self.gamma * max_next_q
                td_error = td_target - self.Q_map[state, action_idx]
                self.Q_map[state, action_idx] += alpha * td_error

                # 5. 상태 이동
                state = next_state

        return self.Q_map


class DQNNetwork(nn.Module):
    def __init__(self,input_size=5,hidden_size=16,output_size=2):    
        super(DQNNetwork,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)    
        x = self.fc2(x)
        return x

class DQN(GridWorld):
    def __init__(self):
        super().__init__()
        self.terminal_state=4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.memory = deque(maxlen=10000)  
        self.batch_size = 16
        
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def state_to_tensor(self, state):
        """상태를 원-핫 벡터로 변환"""
        state_vector = np.zeros(5)
        state_vector[state] = 1
        return torch.FloatTensor(state_vector).to(self.device)    

    def choose_action(self, state):
        """ε-탐욕 정책으로 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)  # 탐험
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()  # 활용
            
    def store_experience(self, state, action_idx, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        """경험 재생으로 학습"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([self.state_to_tensor(s) for s in states])
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([self.state_to_tensor(s) for s in next_states])
        dones = torch.FloatTensor(dones).to(self.device)

        # 현재 Q-값
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 타겟 Q-값
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # 손실 계산 및 업데이트
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def dqn_learning(self, num_episodes=1000, start_state=1):
        """DQN 학습"""
        for episode in range(num_episodes):
            state = start_state
            total_reward = 0
            
            while state != self.terminal_state:
                action_idx = self.choose_action(state)
                action = self.actions[action_idx]
                next_state, penalty = self.get_next_state(state, action)
                reward = self.reward_map[next_state] + penalty
                done = 1 if next_state == self.terminal_state else 0

                self.store_experience(state, action_idx, reward, next_state, done)
                self.train()

                state = next_state
                total_reward += reward

            # Epsilon 감소
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 타겟 네트워크 주기적 업데이트 (예: 10 에피소드마다)
            if episode % 10 == 0:
                self.update_target_network()

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

        # 학습된 Q-값 확인
        q_values = np.zeros((5, 2))
        for s in self.states:
            with torch.no_grad():
                q_values[s] = self.policy_net(self.state_to_tensor(s)).cpu().numpy()
        return q_values        




if __name__ == "__main__":
    
    grid = GridWorld()
    final_value_map = grid.simulate()
    print("최종 가치 함수:", final_value_map)

    td = TemporalDifference()
    q_table = td.q_learning(alpha=0.1, num_episodes=1000, epsilon=0.2, start_state=1)
    print("최종 Q-테이블:")
    print(q_table)
    print("최적 정책 (각 상태에서 선택할 행동):")
    print([td.actions[np.argmax(q_table[s])] for s in td.states])

    dqn = DQN()
    dqn_table = dqn.dqn_learning(num_episodes=1000, start_state=1)
    print("최종 DQN Q-테이블:")
    print(dqn_table)
    print("최적 정책 (각 상태에서 선택할 행동):")
    print([dqn.actions[np.argmax(dqn_table[s])] for s in dqn.states])
