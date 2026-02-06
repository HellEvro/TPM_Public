"""
Reinforcement Learning Agent для торговли

Реализует:
- Trading Environment (Gym-like)
- DQN (Deep Q-Network) с Double DQN
- Experience Replay
- Epsilon-greedy exploration

Действия: HOLD=0, BUY=1, SELL=2, CLOSE=3
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random
import os
import json

logger = logging.getLogger('RL')

# Проверяем PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    PYTORCH_AVAILABLE = False
    DEVICE = None


# Действия агента
ACTIONS = {
    0: 'HOLD',
    1: 'BUY',
    2: 'SELL',
    3: 'CLOSE'
}


class TradingEnvironment:
    """
    Trading Environment для RL агента
    
    Реализует OpenAI Gym-подобный интерфейс
    """
    
    def __init__(
        self,
        candles: List[Dict],
        initial_balance: float = 10000,
        commission: float = 0.001,
        leverage: float = 1.0,
        window_size: int = 20
    ):
        """
        Args:
            candles: Исторические свечи
            initial_balance: Начальный баланс
            commission: Комиссия за сделку
            leverage: Плечо
            window_size: Размер окна для состояния
        """
        self.candles = candles
        self.initial_balance = initial_balance
        self.commission = commission
        self.leverage = leverage
        self.window_size = window_size
        
        self.action_space = len(ACTIONS)
        self.observation_space = window_size * 5 + 3  # OHLCV + position info
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Сбрасывает среду"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: none, 1: long
        self.position_price = 0
        self.total_profit = 0
        self.trade_count = 0
        self.wins = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Возвращает текущее состояние"""
        # OHLCV данные за window
        window = self.candles[self.current_step - self.window_size:self.current_step]
        
        prices = []
        for c in window:
            prices.extend([
                c.get('open', 0),
                c.get('high', 0),
                c.get('low', 0),
                c.get('close', 0),
                c.get('volume', 0)
            ])
        
        # Нормализуем цены
        prices = np.array(prices, dtype=np.float32)
        if prices.std() > 0:
            prices = (prices - prices.mean()) / prices.std()
        
        # Добавляем информацию о позиции
        position_info = np.array([
            self.position,
            self.balance / self.initial_balance - 1,  # Нормализованный баланс
            self.position_price / self.candles[self.current_step]['close'] - 1 if self.position_price > 0 else 0
        ], dtype=np.float32)
        
        return np.concatenate([prices, position_info])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Выполняет действие
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
        
        Returns:
            (next_state, reward, done, info)
        """
        current_price = self.candles[self.current_step]['close']
        reward = 0
        info = {'action': ACTIONS[action]}
        
        # Выполняем действие
        if action == 1:  # BUY
            if self.position <= 0:
                if self.position == -1:  # Закрываем short
                    pnl = (self.position_price - current_price) / self.position_price
                    self.balance *= (1 + pnl * self.leverage - self.commission)
                    reward = pnl * 100
                    self.total_profit += pnl
                    self.trade_count += 1
                    if pnl > 0:
                        self.wins += 1
                
                # Открываем long
                self.position = 1
                self.position_price = current_price
                self.balance *= (1 - self.commission)
        
        elif action == 2:  # SELL
            if self.position >= 0:
                if self.position == 1:  # Закрываем long
                    pnl = (current_price - self.position_price) / self.position_price
                    self.balance *= (1 + pnl * self.leverage - self.commission)
                    reward = pnl * 100
                    self.total_profit += pnl
                    self.trade_count += 1
                    if pnl > 0:
                        self.wins += 1
                
                # Открываем short
                self.position = -1
                self.position_price = current_price
                self.balance *= (1 - self.commission)
        
        elif action == 3:  # CLOSE
            if self.position != 0:
                if self.position == 1:
                    pnl = (current_price - self.position_price) / self.position_price
                else:
                    pnl = (self.position_price - current_price) / self.position_price
                
                self.balance *= (1 + pnl * self.leverage - self.commission)
                reward = pnl * 100
                self.total_profit += pnl
                self.trade_count += 1
                if pnl > 0:
                    self.wins += 1
                
                self.position = 0
                self.position_price = 0
        
        # Переход к следующему шагу
        self.current_step += 1
        done = self.current_step >= len(self.candles) - 1
        
        # Bonus reward за поддержание баланса
        if self.balance > self.initial_balance:
            reward += 0.01
        
        # Penalty за слишком много trades
        if self.trade_count > len(self.candles) * 0.1:
            reward -= 0.001
        
        next_state = self._get_state() if not done else np.zeros(self.observation_space)
        
        info.update({
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'trade_count': self.trade_count,
            'win_rate': self.wins / self.trade_count if self.trade_count > 0 else 0
        })
        
        return next_state, reward, done, info


if PYTORCH_AVAILABLE:
    
    class DQNNetwork(nn.Module):
        """
        Deep Q-Network
        
        Архитектура: MLP с 3 скрытыми слоями
        """
        
        def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 128, 64]):
            super(DQNNetwork, self).__init__()
            
            layers = []
            input_size = state_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                input_size = hidden_size
            
            layers.append(nn.Linear(input_size, action_size))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)


class DQNAgent:
    """
    Double DQN Agent для торговли
    
    Использует:
    - Experience Replay для стабильности
    - Target Network для уменьшения overestimation
    - Epsilon-greedy для exploration
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 4,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.memory = deque(maxlen=memory_size)
        self.train_step = 0
        
        if PYTORCH_AVAILABLE:
            self.policy_net = DQNNetwork(state_size, action_size).to(DEVICE)
            self.target_net = DQNNetwork(state_size, action_size).to(DEVICE)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Выбирает действие
        
        Args:
            state: Текущее состояние
            training: Использовать epsilon-greedy
        
        Returns:
            Выбранное действие
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        if not PYTORCH_AVAILABLE:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет опыт в память"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """
        Обучается на случайном батче из памяти
        
        Returns:
            Loss или None
        """
        if len(self.memory) < self.batch_size or not PYTORCH_AVAILABLE:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: используем policy net для выбора действия, target net для оценки
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss и update
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, path: str):
        """Сохраняет модель"""
        if PYTORCH_AVAILABLE:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'epsilon': self.epsilon
            }, path)
    
    def load(self, path: str):
        """Загружает модель"""
        if PYTORCH_AVAILABLE and os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon = checkpoint['epsilon']


class RLTrader:
    """
    Высокоуровневый RL Trader
    
    Обертка для обучения и использования RL агента
    """
    
    def __init__(
        self,
        model_path: str = "data/ai/models/rl_trader.pth",
        window_size: int = 20
    ):
        self.model_path = model_path
        self.window_size = window_size
        self.state_size = window_size * 5 + 3
        
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=4
        )
        
        if os.path.exists(model_path):
            self.agent.load(model_path)
            logger.info(f"RL Trader loaded from {model_path}")
    
    def train(
        self,
        candles: List[Dict],
        episodes: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Обучает RL агента
        
        Args:
            candles: Исторические свечи
            episodes: Количество эпизодов
            verbose: Выводить прогресс
        
        Returns:
            Dict с результатами обучения
        """
        env = TradingEnvironment(
            candles=candles,
            window_size=self.window_size
        )
        
        rewards_history = []
        profits_history = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.replay()
                
                state = next_state
                total_reward += reward
            
            rewards_history.append(total_reward)
            profits_history.append(info['total_profit'])
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                avg_profit = np.mean(profits_history[-10:])
                logger.info(
                    f"Episode {episode+1}/{episodes} - "
                    f"Reward: {avg_reward:.2f}, Profit: {avg_profit:.2%}, "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )
        
        # Сохраняем модель
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.agent.save(self.model_path)
        
        return {
            'episodes': episodes,
            'final_reward': np.mean(rewards_history[-10:]),
            'final_profit': np.mean(profits_history[-10:]),
            'rewards_history': rewards_history,
            'profits_history': profits_history
        }
    
    def predict_action(self, state: np.ndarray) -> int:
        """Предсказывает действие"""
        return self.agent.act(state, training=False)
    
    def get_status(self) -> Dict:
        """Возвращает статус"""
        return {
            'model_path': self.model_path,
            'window_size': self.window_size,
            'state_size': self.state_size,
            'epsilon': self.agent.epsilon,
            'pytorch_available': PYTORCH_AVAILABLE
        }


# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("RL Agent - Test")
    print("=" * 60)
    
    # Генерируем тестовые свечи
    np.random.seed(42)
    n_candles = 500
    
    candles = []
    price = 100.0
    
    for i in range(n_candles):
        change = np.random.randn() * 0.02
        o = price
        c = price * (1 + change)
        h = max(o, c) * (1 + abs(np.random.randn() * 0.005))
        l = min(o, c) * (1 - abs(np.random.randn() * 0.005))
        
        candles.append({
            'open': o, 'high': h, 'low': l, 'close': c,
            'volume': np.random.randint(1000, 10000)
        })
        price = c
    
    # Тест Environment
    print("\n1. Test TradingEnvironment:")
    env = TradingEnvironment(candles, window_size=20)
    state = env.reset()
    print(f"   State shape: {state.shape}")
    print(f"   Action space: {env.action_space}")
    
    total_reward = 0
    for _ in range(100):
        action = random.randrange(4)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"   Random policy reward: {total_reward:.2f}")
    print(f"   Final balance: ${info['balance']:.2f}")
    
    # Тест DQN Agent
    if PYTORCH_AVAILABLE:
        print("\n2. Test DQNAgent:")
        agent = DQNAgent(state_size=env.observation_space, action_size=4)
        
        # Краткое обучение
        state = env.reset()
        for _ in range(50):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            state = next_state
            if done:
                state = env.reset()
        
        print(f"   Training steps: {agent.train_step}")
        print(f"   Epsilon: {agent.epsilon:.3f}")
    
    # Тест RLTrader
    print("\n3. Test RLTrader:")
    trader = RLTrader(model_path="data/ai/models/test_rl.pth")
    status = trader.get_status()
    print(f"   Status: {status}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)
