# Overview
Traditional Vehicle Routing Problem (VRP) focus mostly on more objective op-timization goals, including minimizing route length, minimizing route time, or over-all workload distribution. Although these traditional models are able to generate op-timized routing plans, many companies have pointed out that these plans are not im-plantation friendly. These objective goals might results in route intersection, route overlap, or other instance that contradict the manual vehicle routing logic. Therefore, this paper has proposed a new subjective consideration, Visual Attractiveness. From a company’s point of view, it is vital for vehicle routing algorithms to generate a more visual attractive routing plan while maintaining a minimum cost.

By utilizing real-time operation data from a garbage recycling firm from Shanxi province, including demand points, fleet information, and facilities, this paper has developed a mathematical model that took visual attractiveness into consideration while fulfilling other operation constraints. 

Furthermore, referencing the savings method and guided local search algorithm, this paper has also proposed a two-phase heuristic algorithm aiming to optimize both the objective goals and visual attractiveness. Through calculating results on multiple dataset and analyzing the results, this paper has verified the reliability and robustness of the proposed algorithm. Moreover, by comparing results and key indicators be-tween the visual attractive routing plan and the traditional routing plan, this paper has validated the necessity of including the visual attractiveness consideration in vehicle routing problems. The proposed method used in this paper is able to increase visual attractiveness of routing plans while maintaining a minimum cost, which can be used as future reference in company vehicle routing related implementation.
# Visualization
## Before:
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/124868e7-fb60-4c70-b09e-ed67397e01f8)
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/e5ab3a3c-c674-4290-87c4-d4a963fdc1ab)
## Conventional Method Optimization:
### Min-Cost:
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/0114d20a-85c1-4516-a071-9c21e859ad20)
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/ea10487a-e30f-4655-bbf2-8c346dabc861)
### C-W Saving Algorithm:
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/3eb4d465-fe40-479c-a262-c8964126d3ba)
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/775f3110-05e0-41cb-b3d5-d9feb93bd874)
## Visual Attractiveness Optimization:
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/5d8ca87e-e778-42f3-94da-34e9fad3da95)
![image](https://github.com/Marcozc19/public.github.io/assets/127183986/f87f3953-bbcd-43bc-a588-ce3ac5c702d9)

