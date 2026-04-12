import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

costs = []
#######################################################
def typing_print(text, speed=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()
#################### BEAUTI FI CATION #################


def predict(x,theeta_0,theeta_1):
    return (x*theeta_1)+ theeta_0

def gradient_descent(x,y,theeta_0 = 0,theeta_1 = 0,learning_rate = 0.01,iterations = 3000):
    m = len(x)
    for _ in range(iterations):
        y_pred = predict(x,theeta_0,theeta_1)
        error = y_pred - y
        theeta_0 = theeta_0 - (learning_rate/m) * np.sum(error)   # "theta = theta - alpha * 1/m * (sum(h(x) - y))"
        theeta_1 = theeta_1 - (learning_rate/m) * np.sum(error*x)
        cost = MSE(x,y, theeta_0, theeta_1)    # calculate the loss after every updation in the theta_0 & theta_1    
        costs.append(cost)

    return theeta_0,theeta_1

def MSE(x,y,theeta_0,theeta_1):                                           
    sample_size = len(x)
    y_pred = predict(x,theeta_0,theeta_1)
    total_error = np.power(y_pred - y,2)
    return (1/sample_size)* np.sum(total_error)


# if __name__ == '__main__':
#     df = pd.DataFrame(data=[
#         [17.3,71.7],[19.3,48.30],[19.5,88.3],[19.7,75.0],[22.9,91.7],
#         [23.1,100.0],[26.4,73.3],[26.8,65.0],[27.6,75.0],[28.1,88.3],
#         [28.2,68.3],[28.7,96.7],[29.0,76.7],[29.6,78.3],[29.9,60.0],
#         [29.9,71.7],[30.3,85.0],[31.3,85.0],[36.0,88.3],[39.5,100.0],
#         [40.4,100.0],[44.3,100.0],[44.6,91.7],[50.4,100.0],[55.9,71.7]
#     ], columns=['Arm_strength','Dynamic_lift'])
#     x=df['Arm_strength']
#     y=df['Dynamic_lift']

#     x_mean, x_std= x.mean(),x.std()
#     y_mean,y_std = y.mean(),y.std()
#     x_scl = (x - x_mean)/x_std
#     y_scl = (y - y_mean)/y_std
#     theeta0,theeta1 = gradient_descent(x_scl,y_scl,0,0,0.001)
#     inp = float(input("Enter Input number: "))
#     inp = (inp - x_mean)/x_std
#     y_pred = predict(inp,theeta0,theeta1)
#     y_pred = (y_pred * y_std) + y_mean
#     print(f"Bias = {theeta0}\nWeight = {theeta1}..")
#     print(f"Predicted Value = {y_pred}\n")
if __name__ == '__main__':
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # 1. Fascinating Intro
    print(f"{BLUE}{BOLD}" + "="*50)
    print("      L I N E A R   R E G R E S S I O N   v1.0")
    print("="*50 + f"{RESET}")

    typing_print(f"{CYAN}Initializing Neural Engine...{RESET}")
    typing_print(f"{CYAN}Loading 'Strength vs Lift' dataset...{RESET}")
    # Your Data
    df = pd.DataFrame(data=[
        [17.3,71.7],[19.3,48.30],[19.5,88.3],[19.7,75.0],[22.9,91.7],
        [23.1,100.0],[26.4,73.3],[26.8,65.0],[27.6,75.0],[28.1,88.3],
        [28.2,68.3],[28.7,96.7],[29.0,76.7],[29.6,78.3],[29.9,60.0],
        [29.9,71.7],[30.3,85.0],[31.3,85.0],[36.0,88.3],[39.5,100.0],
        [40.4,100.0],[44.3,100.0],[44.6,91.7],[50.4,100.0],[55.9,71.7]
    ], columns=['Arm_strength','Dynamic_lift'])
    x, y = df['Arm_strength'], df['Dynamic_lift']
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    # Pre-processing
    x_scl = (x - x_mean) / x_std
    y_scl = (y - y_mean) / y_std
    # Training Animation
    print(f"\n{YELLOW}Training in progress...{RESET}")
    for i in range(10):
        time.sleep(0.1)
        sys.stdout.write(f"\r[{'#' * (i+1)}{' ' * (9-i)}] { (i+1)*10}% Complete")
        sys.stdout.flush()
    print("\n")
    # Train
    theeta0, theeta1 = gradient_descent(x_scl, y_scl, 0, 0, 0.001)
    print(f"{GREEN}✔ Model Trained Successfully!{RESET}")
    print(f"{BOLD}Final Bias (θ0):{RESET} {theeta0:.4f}")
    print(f"{BOLD}Final Weight (θ1):{RESET} {theeta1:.4f}")
    print("-" * 50)
    # Interactive Input
    try:
        val = input(f"\n{YELLOW}Enter Arm Strength to predict Dynamic Lift: {RESET}")
        inp = float(val)

        # Scaling Input
        inp_scaled = (inp - x_mean) / x_std

        # Prediction
        y_pred_scaled = predict(inp_scaled, theeta0, theeta1)
        y_pred = (y_pred_scaled * y_std) + y_mean

        print(f"\n{BLUE}--- PREDICTION RESULT ---{RESET}")
        print(f"For Arm Strength: {BOLD}{inp}{RESET}")
        print(f"Predicted Dynamic Lift: {GREEN}{BOLD}{y_pred:.2f}{RESET}")
        print(f"{BLUE}" + "-" * 25 + f"{RESET}\n")

    except ValueError:
        print(f"{BOLD}\033[91mError: Please enter a valid numerical value.{RESET}")
    

plt.plot(costs)
plt.ylabel("Cost encountered")
plt.xlabel("Iterations")
plt.title("Cost per Iteration")
plt.show()

plt.scatter(x_scl, y_scl)
plt.plot(x_scl, predict(x_scl, theeta0, theeta1))
plt.show()