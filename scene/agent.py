from typing import List, Dict, Any
from queue import Queue

from habitat.core.agent import Agent
from habitat.core.simulator import Observations


class HabitatActuator(Agent): 
    '''
    HabitatActuatorActuator class provides observation and execute the action.
    Cooperate with ProviderPeer to send the observation and receive the action.
    Will be deployed in the computer with the habitat simulator.
    '''

    CHANNELS: List[str] = ["rgb", "depth", "semantic"]
    
    def __init__(self) -> None:
        self.semantic_queue: Queue = None
        self.rgb_queue: Queue = None
        self.depth_queue: Queue = None
        self.state_queue: Queue = None
        self.action_queue: Queue = None
        
    def __transmit_observation(self, observations: Observations) -> None:
        for i, channel in enumerate(self.CHANNELS):
            getattr(self, f"{channel}_queue").put(observations[channel].copy())
            observations[channel] = i
            
        # Convert the Observations object to a dictionary to avoid pickling issues
        state: dict = {key: observations[key] for key in observations.keys()}
        self.state_queue.put(state.copy())        
        
    def __receive_action(self) -> Dict[str, Any]:
        return self.action_queue.get()
        
    def reset(self) -> None:
        pass
    
    def act(self, observations: Observations) -> Dict[str, Any]:
        self.__transmit_observation(observations)
        action: Dict[str, Any] = self.__receive_action()
        return action
        
    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)


class CognitiveUnit: # Not inheriting from Agent on purpose
    '''
    CognitiveUnit class receives observation and generate the action.
    Cooperate with ReceiverPeer to receive the observation and send the action.
    Will be deployed in the workstation. 
    '''
    
    def __init__(self, core_agent: Agent) -> None:
        self.core_agent: Agent = core_agent
        self.observations_queue: Queue = None
        self.action_queue: Queue = None
        
    def reset(self) -> None:
        self.core_agent.reset()
        
    def act(self) -> None:
        observations: Dict[str, Any] = self.observations_queue.get()
        action: Dict[str, Any] = self.core_agent.act(observations)
        self.action_queue.put(action)
        
    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)
        