import torch
from model_class import ActorCriticNetwork

state_dict = torch.load('./input_params/ac_weights.pt', map_location=torch.device('cpu'))

net = ActorCriticNetwork(0.001, [6], 32, 64, 128, 5981)
net.load_state_dict(state_dict)
net.eval()

dummy_input = torch.zeros(1,6)

torch.onnx.export(net, dummy_input, './input_params/actor_critic.onnx', verbose=True)
