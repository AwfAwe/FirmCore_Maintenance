import heapq
from math import sqrt
from objective_function import objective_function, log_objective_function
from collections import deque, Counter
from time import time
from os.path import dirname
from os import getcwd
import copy

sum_of_visited_node_insert = [] 
sum_of_kupdated_node_insert = []
sum_of_visited_node_remove = []
sum_of_kupdated_node_remove = []
sum_of_inserts = 0
sum_of_removes = 0

def FirmCore_Maintenance(multilayer_graph, information, K_map, MCD_map, PCD_map, save=False):
    global sum_of_visited_node_insert
    global sum_of_kupdated_node_insert
    global sum_of_visited_node_remove
    global sum_of_kupdated_node_remove
    global sum_of_inserts
    global sum_of_removes

    layers =  multilayer_graph.get_layers()
    nodes = multilayer_graph.get_nodes()

    NOP = multilayer_graph.number_of_operations

    sum_of_visited_node_insert = [0 for _ in layers]
    sum_of_visited_node_insert.append(0)
    sum_of_kupdated_node_insert = [0 for _ in layers]
    sum_of_kupdated_node_insert.append(0)
    sum_of_visited_node_remove = [0 for _ in layers]
    sum_of_visited_node_remove.append(0)
    sum_of_kupdated_node_remove = [0 for _ in layers]
    sum_of_kupdated_node_remove.append(0)

    start = int(round(time() * 1000))
    print("---------- Operation Start ----------")
    
    Block = int(NOP/100) 
    progress_bar=1
    now = 0 

    for oper in multilayer_graph.operations:
        op, layer, u1, u2 = oper
        if op == 0:
            multilayer_graph.add_edge(u1, u2, layer)
            sum_of_inserts+=1
            for threshold in layers:
                FirmCore_MainT_Insert(multilayer_graph, K_map, MCD_map, PCD_map, u1, u2, layer, threshold)         
        else:
            multilayer_graph.del_edge(u1, u2, layer)
            sum_of_removes+=1
            for threshold in layers:
                FirmCore_MainT_Delete(multilayer_graph, K_map, MCD_map, PCD_map, u1, u2, layer, threshold)         
        now+=1
        if now == Block * progress_bar:
            print("--- Operation Progressing:", progress_bar, "% ---")
            progress_bar+=1
            if progress_bar % 5 == 0:
                end = int(round(time() * 1000))
                print(" > current processing Time: ", (end - start)/1000.00, " (s)\n")

    end = int(round(time() * 1000))
    print("---------- Operation Done ----------")
    print(" >>>> Maintenance Time: ", (end - start)/1000.00, " (s)\n")


def FirmCore_MainT_Insert(MLG, K_map, MCD_map, PCD_map, u1, u2, layer, threshold):

    global sum_of_visited_node_insert
    global sum_of_kupdated_node_insert

    multilayer_graph = MLG.adjacency_list
    nodes = MLG.get_nodes()
    # layers = MLG.layers_iterator
    # print("thr = %d %d %d %d ", threshold, layer, u1, u2)
    # set the root:
    root_u = u1
    if(K_map[u2][threshold]<K_map[u1][threshold]):
        root_u = u2    

    # update the MCD and PCD
    # MCD:
    MCD_u1_is_ok = False
    if(K_map[u1][threshold]<=K_map[u2][threshold]) :
        MCD_map[u1][threshold][layer]+=1
        if(MCD_map[u1][threshold][layer]==K_map[u1][threshold]+1):
            MCD_map[u1][threshold][0]+=1
            if MCD_map[u1][threshold][0] == threshold:
                MCD_u1_is_ok = True
        elif(MCD_map[u1][threshold][layer]==K_map[u1][threshold]):
            MCD_map[u1][threshold][-1]+=1

    MCD_u2_is_ok = False
    if(K_map[u2][threshold]<=K_map[u1][threshold]) :
        MCD_map[u2][threshold][layer]+=1
        if(MCD_map[u2][threshold][layer]==K_map[u2][threshold]+1):
            MCD_map[u2][threshold][0]+=1
            if MCD_map[u2][threshold][0] == threshold:
                MCD_u2_is_ok = True
        elif(MCD_map[u2][threshold][layer]==K_map[u2][threshold]):
            MCD_map[u2][threshold][-1]+=1

    # PCD:
    if K_map[u2][threshold] > K_map[u1][threshold] or K_map[u2][threshold] == K_map[u1][threshold] and MCD_map[u2][threshold][0] >= threshold:
        PCD_map[u1][threshold][layer]+=1
        if(PCD_map[u1][threshold][layer]==K_map[u1][threshold]+1):
            PCD_map[u1][threshold][0]+=1
    
    if K_map[u1][threshold] > K_map[u2][threshold] or K_map[u1][threshold] == K_map[u2][threshold] and MCD_map[u1][threshold][0] >= threshold:
        PCD_map[u2][threshold][layer]+=1
        if(PCD_map[u2][threshold][layer]==K_map[u2][threshold]+1):
            PCD_map[u2][threshold][0]+=1
    
    Insert_layer = layer
    if MCD_u1_is_ok:
        for layer, layer_neighbors in enumerate(multilayer_graph[u1]):
            for neighbor in layer_neighbors:
                if neighbor == u2 and layer==Insert_layer: continue
                if K_map[neighbor][threshold] == K_map[u1][threshold]:
                    PCD_map[neighbor][threshold][layer]+=1
                    if PCD_map[neighbor][threshold][layer]==K_map[u1][threshold]+1:
                        PCD_map[neighbor][threshold][0]+=1

    if MCD_u2_is_ok:
        for layer, layer_neighbors in enumerate(multilayer_graph[u2]):
            for neighbor in layer_neighbors:
                if neighbor == u1 and layer==Insert_layer: continue
                if K_map[neighbor][threshold] == K_map[u2][threshold]:
                    PCD_map[neighbor][threshold][layer]+=1
                    if PCD_map[neighbor][threshold][layer]==K_map[u2][threshold]+1:
                        PCD_map[neighbor][threshold][0]+=1

    CDcreated = [False for node in nodes]
    CDcreated.append(False)
    visited = [False for node in nodes]
    visited.append(False)
    evicted = [False for node in nodes]
    evicted.append(False)
    visited_nodes = set()

    K_thr = K_map[root_u][threshold]
    CD_map = {}
    CD_map[root_u] = copy.deepcopy(PCD_map[root_u][threshold])
    S = deque()
    S.append(root_u)
    visited[root_u] = True
    CDcreated[root_u] = True
    visited_nodes.add(root_u)

    while len(S)!=0:
        cur_u = S.pop()
        if CD_map[cur_u][0]>=threshold :
            for layer, layer_neighbors in enumerate(multilayer_graph[cur_u]):
                for neighbor in layer_neighbors:
                    # if threshold == 2 and layer == 2:
                    #     print(cur_u, neighbor)
                    if not visited[neighbor] and K_map[neighbor][threshold] == K_thr and MCD_map[neighbor][threshold][0] >= threshold:
                        S.append(neighbor)
                        visited_nodes.add(neighbor)
                        visited[neighbor] = True

                        if not CDcreated[neighbor]:
                            CDcreated[neighbor] = True
                            CD_map[neighbor]=copy.deepcopy(PCD_map[neighbor][threshold])
                        else :
                            CD_map[neighbor] =[x + y for x, y in zip(CD_map[neighbor], PCD_map[neighbor][threshold])]
                            CD_map[neighbor][0] = 0
                            for CDs in CD_map[neighbor][1:]:
                                if CDs>K_thr:
                                    CD_map[neighbor][0]+=1

                        sum_of_visited_node_insert[threshold]+=1
                    else:
                        visited[neighbor] = True
                        sum_of_visited_node_insert[threshold]+=1

        else:
            if not evicted[cur_u]:
                Eviction(MLG, cur_u, CD_map, threshold, K_map, CDcreated, evicted)
    
    K_Update_nodes = set()
    Update_nodes = set()
    MCD_Update_nodes = set()

    for node in visited_nodes:
        if visited[node] and not evicted[node]:
            K_map[node][threshold]+=1
            K_Update_nodes.add(node)
            Update_nodes.add(node)

    sum_of_kupdated_node_insert[threshold] += len(K_Update_nodes)

    # delete the relation of the update nodes with adjacency node
    for node in K_Update_nodes:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in K_Update_nodes:
                    MCD_Update_nodes.add(neighbor)
                    Update_nodes.add(neighbor)

    for node in MCD_Update_nodes:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in Update_nodes:
                    if PCD_judge(node, neighbor, threshold, K_map, MCD_map):
                        CD_sub(neighbor, threshold, layer, PCD_map, K_map)

    for node in Update_nodes:
        MCD_map[node][threshold][0] = 0
        MCD_map[node][threshold][-1] = 0
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            MCD_map[node][threshold][layer] = 0
            for neighbor in layer_neighbors:
                if MCD_judge(neighbor, node, threshold, K_map):
                    MCD_map[node][threshold][layer]+=1
            if MCD_map[node][threshold][layer] >= K_map[node][threshold]:
                MCD_map[node][threshold][-1] += 1
                if MCD_map[node][threshold][layer] > K_map[node][threshold]:
                    MCD_map[node][threshold][0] += 1

    for node in Update_nodes:
        PCD_map[node][threshold][0] = 0
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            PCD_map[node][threshold][layer] = 0
            for neighbor in layer_neighbors:
                if PCD_judge(neighbor, node, threshold, K_map, MCD_map):
                    PCD_map[node][threshold][layer]+=1
                # update the 2-hop node
                if neighbor not in Update_nodes:
                    if PCD_judge(node, neighbor, threshold, K_map, MCD_map):
                        CD_add(neighbor, threshold, layer, PCD_map, K_map)
                        # PCD_map[node][threshold][layer]+=1
            if PCD_map[node][threshold][layer] > K_map[node][threshold]:
                PCD_map[node][threshold][0] += 1         
                # if neighbor not in K_Update_nodes


    # if PCD_map[1015][threshold][0]>7:
    #     print(layer, u1, u2, threshold)
# update MCD of u2
def MCD_judge(u1, u2, threshold, K_map):
    return K_map[u1][threshold] >= K_map[u2][threshold] 

# update PCD of u2 
def PCD_judge(u1, u2, threshold, K_map, MCD_map):
    return K_map[u1][threshold] > K_map[u2][threshold] or (K_map[u1][threshold] == K_map[u2][threshold] and MCD_map[u1][threshold][0] >= threshold)

def CD_add(node, threshold, layer, CD_map, K_map):
    CD_map[node][threshold][layer]+=1
    if CD_map[node][threshold][layer]==K_map[node][threshold]+1:
        CD_map[node][threshold][0]+=1

def CD_sub(node, threshold, layer, CD_map, K_map):
    CD_map[node][threshold][layer]-=1
    if CD_map[node][threshold][layer]==K_map[node][threshold]:
        CD_map[node][threshold][0]-=1

def Eviction(MLG, u, CD_map, threshold, K_map, CDcreated, evicted):
    multilayer_graph = MLG.adjacency_list
    evicted[u] = True
    for layer, layer_neighbors in enumerate(multilayer_graph[u]):
        for neighbor in layer_neighbors:
            if K_map[neighbor][threshold] == K_map[u][threshold] :
                if not CDcreated[neighbor]:
                    CDcreated[neighbor] = True
                    CD_map[neighbor]=[0 for _ in range(0,max(MLG.layers_iterator)+1)]
                    # if neighbor==901:
                    #     print("Node***", neighbor, layer, CD_map[neighbor], threshold)

                # if neighbor==901:
                #     print("Node", neighbor, layer, CD_map[neighbor], threshold)
                #     print(CD_map[neighbor][layer])
                CD_map[neighbor][layer]-=1
                if CD_map[neighbor][layer]==K_map[u][threshold]:
                    CD_map[neighbor][0]-=1
                    if CD_map[neighbor][0]==threshold-1 and not evicted[neighbor]:
                        Eviction(MLG, neighbor, CD_map, threshold, K_map, CDcreated, evicted)

def FirmCore_MainT_Delete(MLG, K_map, MCD_map, PCD_map, u1, u2, layer, threshold):
    global sum_of_visited_node_remove
    global sum_of_kupdated_node_remove

    multilayer_graph = MLG.adjacency_list
    nodes = MLG.get_nodes()
    # update the MCD and PCD
    # PCD:
    if PCD_judge(u2, u1, threshold, K_map, MCD_map):
        CD_sub(u1, threshold, layer, PCD_map, K_map)
    
    if PCD_judge(u1, u2, threshold, K_map, MCD_map):
        CD_sub(u2, threshold, layer, PCD_map, K_map)

    # MCD:
    MCD_u1_is_ok = False
    if MCD_judge(u2, u1, threshold, K_map) :
        CD_sub(u1, threshold, layer, MCD_map, K_map)
        if MCD_map[u1][threshold][layer]==K_map[u1][threshold]:
            MCD_u1_is_ok = MCD_map[u1][threshold][0] == threshold-1
        elif MCD_map[u1][threshold][layer]==K_map[u1][threshold]-1:
            MCD_map[u1][threshold][-1]-=1

    MCD_u2_is_ok = False
    if MCD_judge(u1, u2, threshold, K_map):
        CD_sub(u2, threshold, layer, MCD_map, K_map)
        if MCD_map[u2][threshold][layer]==K_map[u2][threshold]:
            MCD_u2_is_ok = MCD_map[u2][threshold][0] == threshold-1
        elif MCD_map[u2][threshold][layer]==K_map[u2][threshold]-1:
            MCD_map[u2][threshold][-1]-=1

    Insert_layer = layer
    if MCD_u1_is_ok:
        for layer, layer_neighbors in enumerate(multilayer_graph[u1]):
            for neighbor in layer_neighbors:
                if neighbor == u2 and layer==Insert_layer: continue
                if K_map[neighbor][threshold] == K_map[u1][threshold]:
                    CD_sub(neighbor, threshold, layer, PCD_map, K_map)

    if MCD_u2_is_ok:
        for layer, layer_neighbors in enumerate(multilayer_graph[u2]):
            for neighbor in layer_neighbors:
                if neighbor == u1 and layer==Insert_layer: continue
                if K_map[neighbor][threshold] == K_map[u2][threshold]:
                    CD_sub(neighbor, threshold, layer, PCD_map, K_map)

    # if threshold == 1:
    #     for node in nodes:
    #         print("DELETE", node, "Kcore:", MCD_map[node][1])

    CDcreated = [False for _ in nodes]
    CDcreated.append(False)
    visited = [False for _ in nodes]
    visited.append(False)

    visited_nodes = set()
    CD_map = {}
    S = deque()

    CD_map[u1] = copy.deepcopy(MCD_map[u1][threshold])
    if CD_map[u1][-1]<threshold:
        S.append(u1)
        visited[u1] = True
        visited_nodes.add(u1)

    CD_map[u2] = copy.deepcopy(MCD_map[u2][threshold])
    if CD_map[u2][-1]<threshold:
        S.append(u2)
        visited[u2] = True
        visited_nodes.add(u2)

    while len(S)!=0:
        cur_u = S.pop()
        for layer, layer_neighbors in enumerate(multilayer_graph[cur_u]):
            for neighbor in layer_neighbors:
                if visited[neighbor] or K_map[neighbor][threshold]>K_map[cur_u][threshold]: continue
                if not CDcreated[neighbor]:
                    CD_map[neighbor] = copy.deepcopy(MCD_map[neighbor][threshold])
                CD_map[neighbor][layer]-=1
                if CD_map[neighbor][layer]==K_map[neighbor][threshold]-1:
                    CD_map[neighbor][-1]-=1
                    if CD_map[neighbor][-1]<threshold:
                        visited[neighbor]=True
                        S.append(neighbor)
                        visited_nodes.add(neighbor)

                sum_of_visited_node_remove[threshold] += 1


    K_Update_nodes = set()
    Update_nodes = set()
    MCD_Update_nodes = set()

    sum_of_kupdated_node_remove[threshold] += len(visited_nodes)

    for node in visited_nodes:
        K_map[node][threshold]-=1
        K_Update_nodes.add(node)
        Update_nodes.add(node)

    # delete the relation of the update nodes with adjacency node
    for node in K_Update_nodes:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in K_Update_nodes:
                    MCD_Update_nodes.add(neighbor)
                    Update_nodes.add(neighbor)

    for node in MCD_Update_nodes:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in Update_nodes:
                    if PCD_judge(node, neighbor, threshold, K_map, MCD_map):
                        CD_sub(neighbor, threshold, layer, PCD_map, K_map)

    for node in Update_nodes:
        MCD_map[node][threshold][0] = 0
        MCD_map[node][threshold][-1] = 0
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            MCD_map[node][threshold][layer] = 0
            for neighbor in layer_neighbors:
                if MCD_judge(neighbor, node, threshold, K_map):
                    MCD_map[node][threshold][layer]+=1

            if MCD_map[node][threshold][layer] >= K_map[node][threshold]:
                MCD_map[node][threshold][-1] += 1
                if MCD_map[node][threshold][layer] > K_map[node][threshold]:
                    MCD_map[node][threshold][0] += 1

    for node in Update_nodes:
        PCD_map[node][threshold][0] = 0
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            PCD_map[node][threshold][layer] = 0
            for neighbor in layer_neighbors:
                if PCD_judge(neighbor, node, threshold, K_map, MCD_map):
                    PCD_map[node][threshold][layer]+=1
                # update the 2-hop node
                if neighbor not in Update_nodes:
                    if PCD_judge(node, neighbor, threshold, K_map, MCD_map):
                        CD_add(neighbor, threshold, layer, PCD_map, K_map)
            if PCD_map[node][threshold][layer] > K_map[node][threshold]:
                PCD_map[node][threshold][0] += 1

def FirmCore_decomposition(multilayer_graph, nodes_iterator, layers_iterator, information, save=False):
    K_map=[[]]
    MCD_map=[[]]
    # MCD_list=[[]]
    PCD_map=[[]]
    
    delta_meta = [{}]
    delta = [{}]
    


    for node in nodes_iterator:
        delta_meta.append({})
        K_map.append([])
        K_map[node].append(0)
        for layer in layers_iterator:
            delta_meta[node][layer] = len(multilayer_graph[node][layer])
        delta_meta[node] = dict(sorted(delta_meta[node].items(), key=lambda d: d[1], reverse=True))

    # print("delta=",delta_meta[1015][7])
    for threshold in layers_iterator:
        delta = copy.deepcopy(delta_meta)
        # print("-------------- threshold = %d --------------"%threshold)
        # delta[node] = {layer1:degree, layer2:degree}
        # set of neighbors that we need to update
        k_max = 0
        k_start = 0
        dist_cores = 0
        for node in nodes_iterator:
            K_map[node].append(list(delta[node].values())[threshold-1])
            k_max = max(k_max, K_map[node][threshold])

        if threshold == 1:
            k_start = 1

        # bin-sort for removing a vertex
        B = [set() for _ in nodes_iterator]

        for node in nodes_iterator:
            B[K_map[node][threshold]].add(node)

        # print("maximum k = ", k_max)
        for k in range(k_start, k_max + 1):
            if B[k]:
                dist_cores += 1
            while B[k]:
                node = B[k].pop()
                K_map[node][threshold] = k
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                    for neighbor in layer_neighbors:
                        if K_map[neighbor][threshold] > k:
                            delta[neighbor][layer] -= 1
                            if delta[neighbor][layer] + 1 == K_map[neighbor][threshold]:
                                neighbors.add(neighbor)

                for neighbor in neighbors:
                    B[K_map[neighbor][threshold]].remove(neighbor)
                    K_map[neighbor][threshold] = heapq.nlargest(threshold, list(delta[neighbor].values()))[-1]
                    B[max(K_map[neighbor][threshold], k)].add(neighbor)

        # print("Number of Distinct cores = %s"%dist_cores)

    # compute MCD [node][shr] = [0:num of >K, layer_1:val_1, layer_2:val_2, ... ] and PCD
    for node in nodes_iterator:
        # vis=[False for node in nodes_iterator]
        MCD_map.append([[]])
        for threshold in layers_iterator:
            # MCD_map[node].append({}) 
            MCD_map[node].append([0 for i in range(0,max(layers_iterator)+2)]) 
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    if K_map[neighbor][threshold]>=K_map[node][threshold]:
                        MCD_map[node][threshold][layer]+=1
                if(MCD_map[node][threshold][layer]>=K_map[node][threshold]):
                    MCD_map[node][threshold][-1]+=1
                    if(MCD_map[node][threshold][layer]>K_map[node][threshold]):
                        MCD_map[node][threshold][0]+=1
            # MCD_map[node][threshold]=dict(sorted(MCD_map[node][threshold].items(), key=lambda d: d[1], reverse=True))

    # compute PCD 
    for node in nodes_iterator:
        # vis=[False for node in nodes_iterator]
        PCD_map.append([[]])
        for threshold in layers_iterator:
            PCD_map[node].append([0 for i in range(0,max(layers_iterator)+1)]) 
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    if K_map[neighbor][threshold] > K_map[node][threshold] or (K_map[neighbor][threshold] == K_map[node][threshold] and MCD_map[neighbor][threshold][0] >= threshold):
                        PCD_map[node][threshold][layer]+=1
                if(PCD_map[node][threshold][layer] > K_map[node][threshold]):
                    PCD_map[node][threshold][0]+=1




    return K_map, MCD_map, PCD_map

def FirmCore(multilayer_graph, information, dataset_name, save=True):
    # out_file = 0

    global sum_of_visited_node_insert
    global sum_of_kupdated_node_insert
    global sum_of_visited_node_remove
    global sum_of_kupdated_node_remove
    global sum_of_inserts
    global sum_of_removes

    try:
        out_file = open( '../output/' + dataset_name + '.txt', "x")
    except:
        out_file = open( '../output/' + dataset_name + '.txt', "w")
    
    
    start = int(round(time() * 1000))
    print("---------- decomposition Start ----------")
    K_map, MCD_map, PCD_map = FirmCore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.get_nodes(), multilayer_graph.get_layers(), information, save)
    print("---------- decomposition Done ----------")
    end = int(round(time() * 1000))
    print(" >>>> decomposition Time: ", (end - start)/1000.00, " (s)\n")
    if save:
        out_file.write("========== Decompositon Processing Time: " + str((end - start)/1000.00)+ " (s) ==========\n")
        # out_file.write("The coreness of nodes after decompositon:\n")
        # for layer in multilayer_graph.get_layers():
        #     out_file.write(">>>> threshold : "+ str(layer) +" <<<<\n")
        #     for node in multilayer_graph.get_nodes():
        #         out_file.write("> node("+ str(node) +")  :  "+str(K_map[node][layer])+"\n")

    
    Maintenance_start = int(round(time() * 1000))
    FirmCore_Maintenance(multilayer_graph, information, K_map, MCD_map, PCD_map, save)
    end = int(round(time() * 1000))
    if save:
        out_file.write("\n\n========== Maintenance Processing Time: " + str((end - Maintenance_start)/1000.00)+ " (s) ==========\n")
        # out_file.write("The coreness of nodes after seriels operations:\n")
        # for layer in multilayer_graph.get_layers():
        #     out_file.write(">>>> threshold : "+ str(layer) +" <<<<\n")
        #     for node in multilayer_graph.get_nodes():
        #         out_file.write("> node("+ str(node) +")  :  " + str(K_map[node][layer])+"\n")
        for layer in multilayer_graph.get_layers():
            out_file.write("\n >>> Average num of visited nodes when threshold is " + str(layer) )

            out_file.write("\n *** Average num of visited nodes for insert case: " + str((sum_of_visited_node_insert[layer])/sum_of_inserts) + "\n")
            out_file.write(" *** Average num of K updates nodes for insert case: " + str((sum_of_kupdated_node_insert[layer])/sum_of_inserts) + "\n")
            out_file.write(" *** Average num of visited nodes for remove case: " + str((sum_of_visited_node_remove[layer])/sum_of_removes) + "\n")
            out_file.write(" *** Average num of K updates nodes for remove case: " + str((sum_of_kupdated_node_remove[layer])/sum_of_removes) + "\n")

        out_file.write("\n *** Overall Processing Time: "+str((end - start)/1000.00) +" (s)\n")

    out_file.close()
        # print(" >>>> decomposition Time: ", (end - start)/1000.00, " (s)\n")