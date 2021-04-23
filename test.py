import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == "__main__":

    file_name = './Models_Uncorrelated_0.0001_0.001/100_average_rewards.list'
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    file_name = './Models_ep101_OU_0.0001_0.001/100_average_rewards.list'
    open_file = open(file_name, "rb")
    loaded_list2 = pickle.load(open_file)
    open_file.close()

    file_name = './Models_ep101_ParameterWithOU_0.0001_0.001/100_average_rewards.list'
    open_file = open(file_name, "rb")
    loaded_list3 = pickle.load(open_file)
    open_file.close()

    print(len(loaded_list))
    print(len(loaded_list2))
    print(len(loaded_list3))
    
    #plt.plot(states, label = "OU")
    plt.plot(loaded_list, label = "Uncorrelated noise")
    plt.plot(loaded_list2, label = "Correlated noise")
    plt.plot(loaded_list3, label = "Parameter noise")
    plt.legend()
    plt.xlabel("Number of episodes")
    plt.ylabel("Episode Reward")
    plt.show()