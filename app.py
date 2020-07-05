import torch
import _pickle as cPickle
import streamlit as st
import numpy as np
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
from model_class import Agent

#Pytorch net params
n_actions = 5981
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_dict = cPickle.load(open('./input_params/action_to_zone_km2_centro.pkl', 'rb'))

# Ensure that load_pg_gan_model is called only once, when the app first loads.
@st.cache(allow_output_mutation=True)
def load_pg_model():
    if torch.cuda.is_available():
        state_dict = torch.load('./input_params/ac_weights.pt')
    else:
        state_dict = torch.load('./input_params/ac_weights.pt', map_location=torch.device('cpu'))

    #Setup policy gradients best agent (scenario 1, 3 layers)
    agent = Agent(alpha=0.001, input_dims=[6], gamma=0.001,
                      n_actions=n_actions)

    #Load network weights
    agent.actor_critic.load_state_dict(state_dict)

    #Set to eval mode
    agent.actor_critic.eval()

    return agent

@st.cache(allow_output_mutation=True)
def read_gdf(path, type):
    if type == 'geojson':
        return gpd.read_file(path, driver='GeoJSON')
    else:
        return gpd.read_file(path)

def get_random_features():
    return np.random.uniform(0,1000, size=(1,6))

def plot_map(gdf):
    map = px.choropleth_mapbox(
        gdf.reset_index(), geojson=gdf.geometry.__geo_interface__, locations='index',
            #color='is_active',
            mapbox_style="carto-positron",
            zoom=11,
            opacity=0.3,
    )
    map.update_layout(mapbox_bearing=0,
                      margin={"r":0,"t":0,"l":0,"b":0})
    return map

def get_agent_action(agent, arr, net):
    inp = torch.tensor(arr).float()
    actions = agent.choose_action(inp)
    reg_action = actions > 0

    agent_actions = np.zeros(net.shape[0])

    for i, v in enumerate(reg_action.view(-1)):
        if action_dict[i] is not None:
            agent_actions[i] = v
        else:
            pass

    return agent_actions

#Setup UI
def main():
    st.title("Streamlit Policy Gradients Regulation Demo")
    st.write('''
    As urban population grows, logistic operations become more complex and increase their costs, especially in the last mile. Cities face challenges when designing their road infrastructure, in particular from a regulation perspective. Done right, restrictions focusing on delivery hours, vehicle sizes or the load they carry should be able to balance the effect of a public policy on profit, people interests and environmental indicators. This paper focuses on generating public policy by training a neural network with reinforcement learning, where an agent represents the regulator and it has control over a simulated urban environment. This agent's task is to maximize a reward, which represents the balance of people, planet and profit. This agent is trained by simulating a commercial area in Lima, Peru, and comparing the model's proposal to how the city currently regulates this area. The results point to an interesting insight into where and how regulations should be constructed and a possible decision support system for authorities. 
    ''')
    st.write("On the plot, green lanes allow circulation of trucks, yellow lanes do not")

    # Read in models from the data files.
    agent = load_pg_model()

    #load network
    net = read_gdf('./input_params/downtown_network', None)

    net['is_active'] = 1

    st.sidebar.title('Normalized Parameters')
    CO2 = st.sidebar.slider('CO2', min_value=0.00, max_value=1.00, value=0.50)
    CO = st.sidebar.slider('CO', min_value=0.00, max_value=1.00, value=0.50)
    NOx = st.sidebar.slider('NOx', min_value=0.00, max_value=1.00, value=0.50)
    PMx = st.sidebar.slider('PMx', min_value=0.00, max_value=1.00, value=0.50)
    Noise = st.sidebar.slider('Noise', min_value=0.00, max_value=1.00, value=0.50)
    Fuel = st.sidebar.slider('Fuel', min_value=0.00, max_value=1.00, value=0.50)

    input_array = [CO2, CO, NOx, PMx, Noise, Fuel]

    # for case in sanity_checks:
    #     a = get_agent_action(agent, case, net)
    #     print(sum(a))

    action = get_agent_action(agent, input_array, net)
    action = action[:net.shape[0]]

    net['is_active'] = action

    net.plot(column='is_active', legend=True, cmap='Set3', categorical=True)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    st.pyplot()

    st.write("Total lanes", net.shape[0])
    st.write("Regulated lanes", net['is_active'].sum())

if __name__ == "__main__":
    main()
