import copy
import sys
import time
import argparse
import os
from queue import Queue, PriorityQueue


class Node:
    # Class Constructor
    # Node (info, parent, cost, h, pathInfo)
    graph = None

    def __init__(self, info: list, parent, cost=0, heuristic=0, path_info={}):
        """ type: list(dictionary) - contains informations about the water pots """
        self.info = info

        """ Node - is the current node's parent in the path """
        self.parent = parent

        """ type: number - is the cost of the path to the current node """
        self.cost = cost

        """ number - is the cost of the path from the current node to the destination node """
        self.heuristic = heuristic

        """ type: number - is cost + heuristic """
        self.estimated_cost = self.cost + self.heuristic

        """type: type: dictionary- contains informations about the combanation of the pots"""
        self.path_info = path_info

    def __str__(self):
        result_string = ""

        # { "id" : number, "capacity" : number, "quantity" : number, "color" : string }
        for pot in self.info:
            values = list(pot.values())
            if values[3] != None and values[3] != "undefined":
                result_string += f"Pot -> id = {values[0]}; capacity = {values[1]}; quantity = {values[2]};  color = {values[3]};\n"
            else:
                result_string += f"Pot -> id = {values[0]}; capacity = {values[1]}; quantity = {values[2]};\n"

        return result_string

    def __repr__(self):
        result_string = f"Node info:" + "\n"
        for pot in self.info:
            result_string += str(pot) + "\n"
        if self.parent != None:
            result_string += f"Parent: {str(self.parent.info)}" + "\n"
        else:
            result_string += f"Parent: None" + "\n"
        result_string += f"Cost: {str(self.cost)}" + "\n"
        result_string += f"Heuristic: {str(self.heuristic)}" + "\n"
        result_string += f"Estimated Cost: {str(self.estimated_cost)}" + "\n"
        result_string += f"Path Info: {str(self.path_info)}" + "\n"
        return result_string

    def __lt__(self, obj):
        if self.estimated_cost < obj.estimated_cost:
            return 1
        if self.estimated_cost == obj.estimated_cost and self.cost >= obj.cost:
            return 1
        return 0

    def compare_estimated_cost(self, node):
        if self.estimated_cost >= node.estimated_cost:
            return True
        return False

    def get_path(self):
        node = self
        path = [node]

        while node.parent is not None:
            path.insert(0, node.parent)
            node = node.parent

        return path

    def print_path(self, print_cost=False, print_length=False):
        result_string = ""
        path = self.get_path()

        for node in path:
            path_info = node.path_info
            if path_info != None and node.parent is not None:
                values = list(path_info.values())
                # { "first_pot" : potInfo, "second_pot" : potInfo, "color" : string, "poured_liters" : number }
                result_string += f"{values[3]} liters of colored water {values[2]} were poured from the pot {values[0]} into the pot {values[1]}" + "\n"
                result_string += node.__str__() + "\n"

        if print_cost:
            result_string += "Path cost: " + str(self.cost) + "\n"
        if print_length:
            result_string += "Path length: " + str(len(path)) + "\n"

        return result_string

    def contain_path(self, checked_node):
        node = self

        while node is not None:
            if node.info == checked_node:
                return True
            node = node.parent

        return False

    def has_final_state(self, final_state):
        for (quantity, color) in final_state:
            ok = False
            for pot in self.info:
                # { "id" : number, "capacity" : number, "quantity" : number, "color" : string }
                pot_values = list(pot.values())
                if quantity == pot_values[2] and color == pot_values[3]:
                    ok = True
                    break
            if not ok:
                return ok
        return True


class Graph:
    def __init__(self, input_filename, output_file, timeout):
        self.timeout = timeout

        self.start_time = time.time()

        self.output_file = output_file

        try:
            reader = open(input_filename, "r")

            try:
                file_content = reader.read()

                initial_info, start_state, final_state = self.split_file(file_content)

                self.color_combinations, self.color_cost = self.parse_initial_info(initial_info)

                self.start_state = self.parse_start_state(start_state)

                self.final_state = self.parse_final_state(final_state)

            except:
                print(f"Could not parse the file! Filename:{input_filename}", file=self.output_file)
                sys.exit(0)
        except:
            print(f"Could not open the input file! Filename:{input_filename}", file=self.output_file)
            sys.exit(0)

    def check_timeout(self, current):
        current_time = time.time()

        time_difference = current_time - current
        time_difference *= 1000

        if self.timeout < time_difference:
            print(f"Time Limit Exceeded", file=self.output_file)
            return True

        return False

    def split_file(self, file_content):

        first_split = file_content.strip().split("start_state")
        second_split = first_split[1].strip().split("final_state")

        initial_info = first_split[0].strip().split('\n')
        start_state = second_split[0].strip().split('\n')
        final_state = second_split[1].strip().split('\n')

        return initial_info, start_state, final_state

    def parse_initial_info(self, initial_info):

        color_combinations = list()
        color_cost = dict()

        for line in initial_info:
            content = line.split()
            if len(content) == 2:
                color_cost[content[0]] = int(content[1])
            else:
                color_combinations.append((content[0], content[1], content[2]))
        return color_combinations, color_cost

    def parse_start_state(self, start_state):

        state = []
        count = 0
        for line in start_state:
            content = line.split()
            if len(content) == 2:
                new_dict = dict()
                new_dict = {"id": count, "capacity": int(content[0]), "quantity": 0, "color": None}
                state.append(new_dict)
            else:
                if content[2] not in self.color_cost.keys():
                    raise Exception
                new_dict = dict()
                new_dict = {"id": count, "capacity": int(content[0]), "quantity": int(content[1]), "color": content[2]}
                state.append(new_dict)
            count += 1
        return state

    def parse_final_state(self, final_state):

        state = []
        for line in final_state:
            content = line.split()
            if content[1] not in self.color_cost.keys():
                raise Exception
            state.append((int(content[0]), content[1]))
        return state

    def find_distinct_colors(self):

        found_colors = set()
        for (color1, color2, color3) in self.color_combinations:
            found_colors.add(color1)
            found_colors.add(color2)
        return found_colors

    def count_combinations(self, info):

        found_colors = self.find_distinct_colors()
        number = 0
        for color in found_colors:
            for pot in info:
                values = list(pot.values())
                # "Pot -> id = {values[0]}; color = {values[3]}; quantity = {values[2]}; capacity = {values[1]}\n"
                if values[3] == color:
                    number += 1
                    break
        return number

    def count_final(self, info):

        number = 0
        for (color1, color2, color3) in self.color_combinations:
            for pot in info:
                values = list(pot.values())
                # "Pot -> id = {values[0]}; color = {values[3]}; quantity = {values[2]}; capacity = {values[1]}\n"
                if values[3] == color3:
                    number += 1
        return number

    def test_node(self, info):
        number = 0

        for (quantity, color) in self.final_state:
            for pot in info:
                values = list(pot.values())
                if values[3] == color and values[2] == quantity:
                    number += 1
                    break
        if len(self.final_state) == number:
            return True
        return False

    def check_final(self, info):

        for (quantity, color) in self.final_state:
            count_errors = 0
            for pot in info:
                #                 print(pot)
                values = list(pot.values())
                if quantity > values[1]:
                    count_errors += 1
                if count_errors == len(info):
                    return False
        return True

    def check_node(self, info):

        count_final_colors = self.count_final(info)
        count_color_combinations = self.count_combinations(info)
        check_final_state = self.check_final(info)

        if check_final_state == False:
            return False

        if count_final_colors == len(self.color_combinations) and count_color_combinations == 0:
            return True
        elif count_final_colors == 0 and count_color_combinations >= 2:
            return True
        elif count_color_combinations != 0 and count_final_colors != 0:
            return True

        return False

    def check_color_combination(self, first_color, second_color):

        for (color1, color2, color3) in self.color_combinations:
            if ((color1 == first_color) and (color2 == second_color)) or (
                    (color1 == second_color) and (color2 == first_color)):
                return True, color3
        return False, None

    def generate_succesors(self, node, heuristic = "default"):

        succesors = list([])
        count = 0
        for first_pot in node.info:
            # { "id" : number, "capacity" : number, "quantity" : number, "color" : string }
            first_pot_values = list(first_pot.values())

            number = 0
            for second_pot in node.info:
                if count == number:
                    number += 1
                    continue

                second_pot_values = list(second_pot.values())
                new_node = copy.deepcopy(node.info)

                first_pot_copy = copy.deepcopy(first_pot_values)
                second_pot_copy = copy.deepcopy(second_pot_values)

                quantity_difference = second_pot_copy[1] - second_pot_copy[2]
                transfer_cost = 0
                color_path = dict()

                #  { "first_pot" : potInfo, "second_pot" : potInfo, "color" : string, "poured_liters" : number }
                color_path["first_pot"] = first_pot_copy[0]  # take the id of the pots we use
                color_path["second_pot"] = second_pot_copy[0]

                if quantity_difference > 0:  # the second pot is not full
                    if second_pot_copy[2] == 0:  # second pot is empty
                        second_pot_copy[3] = first_pot_copy[3]
                        color_path["color"] = first_pot_copy[3]

                        if first_pot_copy[3] in self.color_cost.keys():  # this color is not undefined
                            transfer_cost += quantity_difference * self.color_cost[first_pot_copy[3]]
                        if transfer_cost == 0:
                            transfer_cost += quantity_difference * 1

                    else:  # the second pot is not empty, so we are making a new color (and we need to take into consideration the color already found there)
                        if first_pot_copy[3] != None and first_pot_copy[3] == second_pot_copy[3]:
                            # we found the same color in the pots and the color is not undefined
                            color_path["color"] = first_pot_copy[3]

                            if first_pot_copy[3] in self.color_cost.keys():  # this color is not undefined
                                transfer_cost += quantity_difference * self.color_cost[first_pot_copy[3]]
                        else:

                            checker, color = self.check_color_combination(first_pot_copy[3], second_pot_copy[3])
                            if checker:
                                second_pot_copy[3] = color
                                color_path["color"] = first_pot_copy[3]
                                if first_pot_copy[3] in self.color_cost.keys():  # this color is not undefined
                                    transfer_cost += quantity_difference * self.color_cost[first_pot_copy[3]]

                            elif first_pot_copy[2] > 0:  # first pot is not empty and the resulted color is undefined
                                # color cost = quantity diff *  first color cost + quantity found in second pot * second color cost
                                first_cost = second_cost = 0

                                if first_pot_copy[3] in self.color_cost.keys():  # this color is not undefined
                                    first_cost = quantity_difference * self.color_cost[first_pot_copy[3]]

                                if second_pot_copy[3] in self.color_cost.keys():  # this color is not undefined
                                    second_cost = second_pot_copy[2] * self.color_cost[second_pot_copy[3]]

                                if first_cost == 0:  # first color is undefined
                                    color_path["color"] = "undefined"
                                    transfer_cost += quantity_difference * 1
                                elif second_cost == 0:  # the color is already undefined
                                    color_path["color"] = "undefined"
                                    transfer_cost += second_pot_copy[2] * 1
                                else:
                                    color_path["color"] = first_pot_copy[3]
                                    transfer_cost += first_cost + second_cost
                                second_pot_copy[3] = None

                    if quantity_difference > first_pot_copy[2]:
                        # we can move more liquid than we have => the first pot will be empty
                        second_pot_copy[2] += first_pot_copy[2]
                        color_path["poured_liters"] = first_pot_copy[2]
                        first_pot_copy[2] = 0
                    else:
                        color_path["poured_liters"] = quantity_difference
                        first_pot_copy[2] -= quantity_difference
                        second_pot_copy[2] += quantity_difference

                if first_pot_copy[2] == 0:
                    first_pot_copy[3] = None
                if second_pot_copy[2] == 0:
                    second_pot_copy[3] = None

                c = 0
                for new_pot in new_node:
                    # "Pot -> id = {values[0]}; color = {values[3]}; quantity = {values[2]}; capacity = {values[1]}\n"
                    if c == count:
                        new_pot_values = list(new_pot.values())
                        new_pot_values[2] = copy.deepcopy(first_pot_copy[2])
                        new_pot_values[3] = copy.deepcopy(first_pot_copy[3])
                        new_dict = {"id": new_pot_values[0], "capacity": new_pot_values[1],
                                    "quantity": new_pot_values[2], "color": new_pot_values[3]}

                        new_node[c] = copy.deepcopy(new_dict)
                    elif c == number:
                        new_pot_values = list(new_pot.values())
                        new_pot_values[2] = copy.deepcopy(second_pot_copy[2])
                        new_pot_values[3] = copy.deepcopy(second_pot_copy[3])
                        new_dict = {"id": new_pot_values[0], "capacity": new_pot_values[1],
                                    "quantity": new_pot_values[2], "color": new_pot_values[3]}
                        new_node[c] = copy.deepcopy(new_dict)
                    c += 1

                if (not node.contain_path(new_node)) and self.check_node(new_node):
                    # (self, info: list, parent,  cost = 0, heuristic = 0, path_info = {})
                    #                         print(f"count : {count} number: {number}")
                    p = copy.deepcopy(color_path)
                    n = Node(new_node, node, node.cost + transfer_cost, self.compute_heuristic(new_node, heuristic), p)
                    # print(self.compute_heuristic(new_node,heuristic))
                    succesors.append(n)
                number += 1
            count += 1

        return succesors

    def find_color(self, color, info):
        for pot in info:
            values = list(pot.values())
            if values[3] == color:
                return True
        return False

    def count_color_apperances(self, color, info):
        number = 0
        for pot in info:
            values = list(pot.values())
            if values[3] == color:
                number += 1
        return number

    def compute_heuristic(self, node_info, heuristic):


        # default heuristic
        if heuristic == "default":
            if self.check_final(node_info):
                return 0
            return 1
        elif heuristic == "admissible 1":

            # compunte the number of transfers required to get the colors specified in final states
            # the heuristic is the minimum number
            # for pots that are already in final state, the heuristic equals to 0

            heuristics = list()

            for (quantity, color) in self.final_state:
                current_heuristic = 0

                if not self.find_color(color, node_info):
                    for (color1, color2, color3) in self.color_combinations:
                        if color3 == color:
                            cost_color1 = self.color_cost[color1]
                            cost_color2 = self.color_cost[color2]

                            color1_apperances = self.count_color_apperances(color1, node_info)
                            color2_apperances = self.count_color_apperances(color2, node_info)

                            color1_total = cost_color1 * color1_apperances
                            color2_total = cost_color2 * color2_apperances

                            current_heuristic = min(color1_total, color2_total)

                heuristics.append(current_heuristic)

            return min(heuristics)

        elif heuristic == "admissible 2":

            # for pots that are already in final state, the heuristic equals to 0
            # we compute the minimum cost between the colors that can be mixed in order to get a final state color

            heuristics = list()
            for (quantity, color) in self.final_state:
                current_heuristic = 0
                if not self.find_color(color, node_info):
                    for (color1, color2, color3) in self.color_combinations:
                        if color3 == color:
                            cost_color1 = self.color_cost[color1]
                            cost_color2 = self.color_cost[color2]
                            current_heuristic = min(cost_color1, cost_color2)
                heuristics.append(current_heuristic)
            return min(heuristics)

        elif heuristic == "inadmissible":

            # an inadmissible heuristic can be created by changing the second admissible heuristic
            # instead of taking the minimum between the costs, we can take the maximum value
            # and to be sure it is inadmissible, we take the square value

            heuristics = list()

            for (quantity, color) in self.final_state:
                current_heuristic = 0

                if not self.find_color(color, node_info):
                    for (color1, color2, color3) in self.color_combinations:
                        cost_color1 = self.color_cost[color1]
                        cost_color2 = self.color_cost[color2]

                        current_heuristic = max(cost_color1 * cost_color1, cost_color2 * cost_color2)

                heuristics.append(current_heuristic)

            return max(heuristics)


def breadth_first(graph,  num_sols=1):
    print("***************************************** BFS Algorithm *****************************************",
          file=graph.output_file)
    start_time = time.time()
    start_node = Node(graph.start_state, None, 0, graph.compute_heuristic(graph.start_state, "default"), {})

    print("------------------------- Start State -------------------------", file=graph.output_file)
    print(start_node, file=graph.output_file)
    print("---------------------------------------------------------------\n", file=graph.output_file)

    q = Queue()
    q.put(start_node)
    max_number_nodes = 1
    number_computed_nodes = 1

    if graph.check_timeout(start_time):
        return

    if start_node.has_final_state(graph.final_state):
        print("The start state is the same as a final state", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print("Path cost: " + str(start_node.cost) + "\n", file=graph.output_file)
        print("Path length: " + str(1) + "\n", file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        return

    if (graph.check_node(start_node.info)) == False:
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)

    while not q.empty():
        if graph.check_timeout(start_time):
            return
        current_node = q.get()
        next_states_list = graph.generate_succesors(current_node, "default")
        number_computed_nodes += len(next_states_list)

        for state in next_states_list:
            if graph.test_node(state.info):
                current_time = time.time()
                time_difference = current_time - graph.start_time
                time_difference *= 1000
                print(state.print_path(print_cost=True, print_length=True), file=graph.output_file)

                if graph.check_timeout(start_time):
                    return

                print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
                print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
                print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

                print("----------------------------------------------------------\n", file=graph.output_file)
                num_sols -= 1
                if num_sols == 0:
                    print(
                        "***************************************** End BFS Algorithm *****************************************",
                        file=graph.output_file)
                    return
            q.put(state)
        max_number_nodes = max(max_number_nodes, q.qsize())

    print("***************************************** End BFS Algorithm *****************************************",
              file=graph.output_file)
    return "Finished"


def depth_first(graph, num_sols=1):

    print(
        "***************************************** DFS Algorithm *****************************************",
        file=graph.output_file)
    start_time = time.time()
    start_node = Node(graph.start_state, None, 0, graph.compute_heuristic(graph.start_state, "default"), {})
    max_number_nodes = 1
    number_computed_nodes = 1

    print("------------------------- Start State -------------------------", file=graph.output_file)
    print(start_node, file=graph.output_file)
    print("---------------------------------------------------------------\n", file=graph.output_file)

    if graph.check_timeout(start_time):
        return

    if start_node.has_final_state(graph.final_state):
        print("The start state is the same as a final state", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print("Path cost: " + str(start_node.cost) + "\n", file=graph.output_file)
        print("Path length: " + str(1) + "\n", file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        return


    if (graph.check_node(start_node.info)) == False:
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)
        return

    df(start_time, max_number_nodes, number_computed_nodes, graph, start_node, num_sols)

    print(
        "***************************************** End DFS Algorithm *****************************************",
        file=graph.output_file)

    return "Finished"


def df(start_time, max_number_nodes, num_of_computed_nodes, graph, current_node, num_sols=1):

    if num_sols <= 0:
        return num_sols
    max_number_nodes += 1

    if graph.test_node(current_node.info):
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        if graph.check_timeout(start_time):
            return -1

        print(current_node.print_path(print_cost=True, print_length=True), file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        print("----------------------------------------------------------\n", file=graph.output_file)
        num_sols -= 1
        if num_sols == 0:
            return num_sols
    else:
        if graph.check_timeout(start_time):
            return -1
        successors = graph.generate_succesors(current_node, "default")
        num_of_computed_nodes += len(successors)
        max_number_nodes = max(max_number_nodes, len(successors))
        for sc in successors:
            if num_sols != 0:
                num_sols = df(start_time, max_number_nodes, num_of_computed_nodes, graph, sc, num_sols)
    return num_sols


def iterative_depth_first(graph, num_sol=1):
    print(
        "***************************************** IDF Algorithm *****************************************",
        file=graph.output_file)
    start_time = time.time()
    max_number_nodes = 1
    num_of_computed_nodes = 1
    i = 1
    start_node = Node(graph.start_state, None, 0, graph.compute_heuristic(graph.start_state, "default"), {})

    print("------------------------- Start State -------------------------", file=graph.output_file)
    print(start_node, file=graph.output_file)
    print("---------------------------------------------------------------\n", file=graph.output_file)

    if graph.check_timeout(start_time):
        return

    if start_node.has_final_state(graph.final_state):
        print("The start state is the same as a final state", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print("Path cost: " + str(start_node.cost) + "\n", file=graph.output_file)
        print("Path length: " + str(1) + "\n", file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        return

    if not (graph.check_node(start_node.info)):
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)
        return

    while num_sol != 0:
        if graph.check_timeout(start_time):
            return
        num_sol = idf(start_time, max_number_nodes, num_of_computed_nodes, graph, start_node, i, num_sol)
        i += 1
    print(
        "***************************************** End IDF Algorithm *****************************************",
        file=graph.output_file)
    return "Finished"


def idf(start_time, max_number_nodes, num_of_computed_nodes, graph, current_node, height, num_sol=1):
    num_of_computed_nodes += 1

    if height == 1 and graph.test_node(current_node.info):
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000
        print(current_node.print_path(print_cost=True, print_length=True), file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        print("----------------------------------------------------------\n", file=graph.output_file)
        num_sol -= 1
        if num_sol == 0:
            return num_sol
    if height > 1:

        successors = graph.generate_succesors(current_node, "default")
        num_of_computed_nodes += len(successors)
        max_number_nodes = max(max_number_nodes, len(successors))
        for sc in successors:
            if num_sol != 0:
                num_sol = idf(start_time, max_number_nodes, num_of_computed_nodes, graph, sc, height - 1, num_sol)
    return num_sol


def a_star(graph, num_sol=1, heuristic="default"):
    print("***************************************** A Star Algorithm *****************************************",
          file=graph.output_file)


    q = PriorityQueue()
    start_node = Node(graph.start_state, None, 0, graph.compute_heuristic(graph.start_state, "default"), {})
    q.put(start_node)
    start_time = time.time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    print("------------------------- Start State -------------------------", file=graph.output_file)
    print(start_node, file=graph.output_file)
    print("---------------------------------------------------------------\n", file=graph.output_file)

    if start_node.has_final_state(graph.final_state):
        print("The start state is the same as a final state", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print("Path cost: " + str(start_node.cost) + "\n", file=graph.output_file)
        print("Path length: " + str(1) + "\n", file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_num_of_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        return "Finished"

    if (graph.check_node(start_node.info)) == False:
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_num_of_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)

    if graph.check_timeout(start_time):
        return

    while not q.empty():
        current_node = q.get()
        if graph.test_node(current_node.info):

            if graph.check_timeout(start_time):
                return

            current_time = time.time()
            time_difference = current_time - graph.start_time
            time_difference *= 1000
            print("------------------------- Solution -------------------------", file=graph.output_file)
            print(current_node.print_path(print_cost=True, print_length=True), file=graph.output_file)

            print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
            print(f"Maximum number of nodes in memory: {max_num_of_nodes}\n", file=graph.output_file)
            print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

            print("----------------------------------------------------------\n", file=graph.output_file)
            num_sol -= 1
            if num_sol == 0:
                return "Finished"

        successors = graph.generate_succesors(current_node, heuristic)
        num_of_computed_nodes += len(successors)

        if graph.check_timeout(start_time):
            return

        for successor in successors:
            if any(successor.info == item.info for item in q.queue):
                continue
            q.put(successor)
        max_num_of_nodes = max(max_num_of_nodes, q.qsize())


def a_star_opt(graph, heuristic="default"):

    print("\n***************************************** A Star Opt Algorithm *****************************************", file=graph.output_file)
    unexpanded = list()
    expanded = list()
    node = Node(graph.start_state, None, 0, graph.compute_heuristic(graph.start_state, heuristic), {})
    unexpanded.append(node)
    start_time = time.time()
    max_number_nodes = 1
    number_computed_nodes = 1

    if graph.check_timeout(start_time):
        return

    print("------------------------- Start State -------------------------", file=graph.output_file)
    print(node, file=graph.output_file)
    print("---------------------------------------------------------------\n", file=graph.output_file)

    if node.has_final_state(graph.final_state):
        print("The start state is the same as a final state", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print("Path cost: " + str(node.cost) + "\n", file=graph.output_file)
        print("Path length: " + str(1) + "\n", file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        return "Finished"

    if not (graph.check_node(node.info)):
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)
        return "Finished"

    found = False
    #     succesors = graph.generate_succesors(node, heuristic)
    #     print(succesors)
    while len(unexpanded) > 0:

        if graph.check_timeout(start_time):
            return

        if max_number_nodes < len(unexpanded):
            max_number_nodes = len(unexpanded)

        current = unexpanded.pop(0)
        expanded.append(current)

        if graph.test_node(current.info):
            print("------------------------- Solution -------------------------", file=graph.output_file)
            found = True

            current_time = time.time()
            time_difference = current_time - graph.start_time
            time_difference *= 1000

            print(current.print_path(print_cost=True, print_length=True), file=graph.output_file)

            print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
            print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
            print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

            print("----------------------------------------------------------\n", file=graph.output_file)


            print("***************************************** End A Star Opt Algorithm *****************************************",
                    file=graph.output_file)
            return "Finished"

        succesors = graph.generate_succesors(current, heuristic)
        number_computed_nodes += len(succesors)

        if graph.check_timeout(start_time):
            return

        for succesor in succesors:
            ok = False
            for elem in unexpanded:
                if succesor.info == elem.info:
                    ok = True
                    if succesor.compare_estimated_cost(elem):
                        if succesor in succesors:
                            succesors.remove(succesor)
                    else:
                        if elem in unexpanded:
                            unexpanded.remove(elem)

            if not ok:
                for elem in expanded:
                    if succesor.info == elem.info:
                        if succesor.compare_estimated_cost(elem):
                            if succesor in succesors:
                                succesors.remove(succesor)
                        else:
                            if elem in unexpanded:
                                unexpanded.remove(elem)

        for succesor in succesors:
            position = 0
            counter = 0
            for element in unexpanded:
                if element.estimated_cost > succesor.estimated_cost or (element.estimated_cost == succesor.estimated_cost and element.cost < succesor.cost):
                    position = counter
                    break
            if position != 0:
                unexpanded.insert(position, succesor)
            else:
                unexpanded.append(succesor)


    if not found:
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)

        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_number_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {number_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)

    print("***************************************** End A Star Opt Algorithm *****************************************",
          file=graph.output_file)

    if graph.check_timeout(start_time):
        return


def ida_star(graph, num_sol=1, heuristic='default'):

    print(
        "***************************************** IDA Star Algorithm *****************************************",
        file=graph.output_file)
    start_node = Node(graph.start_state, None, 0, graph.compute_heuristic(graph.start_state, heuristic), {})
    limit = start_node.estimated_cost
    start_time = time.time()
    num_of_computed_nodes = 1
    max_num_of_nodes = 1

    if graph.check_timeout(start_time):
        return

    print("------------------------- Start State -------------------------", file=graph.output_file)
    print(start_node, file=graph.output_file)
    print("---------------------------------------------------------------\n", file=graph.output_file)

    if start_node.has_final_state(graph.final_state):
        print("The start state is the same as a final state", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print("Path cost: " + str(start_node.cost) + "\n", file=graph.output_file)
        print("Path length: " + str(1) + "\n", file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_num_of_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        return "Finished"

    if (graph.check_node(start_node.info)) == False:
        print("------------------------- Input Without Solution -------------------------", file=graph.output_file)
        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_num_of_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {num_of_computed_nodes}\n", file=graph.output_file)

        print("------------------------------------------------------------------------\n", file=graph.output_file)

    while True:
        if graph.check_timeout(start_time):
            return

        _, result = construct_path(graph, start_node, limit, num_sol, start_time, heuristic,
                                   num_of_computed_nodes, max_num_of_nodes)
        limit = result
        if result == "done":
            print(
                "***************************************** End IDA Star Algorithm *****************************************",
                file=graph.output_file)
            return "Finished"

        if result == float('inf'):
            print("No more solutions", file=graph.output_file)
            print(
                "***************************************** End IDA Star Opt Algorithm *****************************************",
                file=graph.output_file)
            return "Finished"


def construct_path(graph, current_node, limit, num_sol, start_time, heuristic, comp_nodes, max_nodes):

    if graph.check_timeout(start_time):
        return 0, "done"

    if current_node.estimated_cost > limit:
        return num_sol, current_node.estimated_cost

    if graph.test_node(current_node.info):
        print("------------------------- Solution -------------------------", file=graph.output_file)

        current_time = time.time()
        time_difference = current_time - graph.start_time
        time_difference *= 1000

        print(current_node.print_path(print_cost=True, print_length=True), file=graph.output_file)

        print(f"Time elapsed since starting the program: {time_difference}\n", file=graph.output_file)
        print(f"Maximum number of nodes in memory: {max_nodes}\n", file=graph.output_file)
        print(f"Number of expanded nodes: {comp_nodes}\n", file=graph.output_file)

        print("----------------------------------------------------------\n", file=graph.output_file)

        num_sol -= 1
        if num_sol == 0:
            return 0, "done"

    successors = graph.generate_succesors(current_node, heuristic)
    comp_nodes += len(successors)
    max_nodes = max(max_nodes, len(successors))
    mini = float('inf')

    if graph.check_timeout(start_time):
        return 0, "done"

    for successor in successors:
        num_sol, res = construct_path(graph, successor, limit, num_sol, start_time, heuristic,
                                      comp_nodes, max_nodes)

        if res == "done":
            return 0, "done"
        if res < mini:
            mini = res

    if graph.check_timeout(start_time):
        return 0, "done"

    return num_sol, mini


def create_input_file():
    input_text = """
    albastru galben verde
    albastru 3
    galben 7
    verde 10
    start_state
    5 3 albastru
    4 2 galben
    3 0
    5 5 galben
    final_state
    4 verde
    3 galben
    """

    with open("input/input3.txt", "w+") as fin:
        fin.write(input_text)


def argument_reader():
    argument_parser = argparse.ArgumentParser(description="Water Pots Problem Solver")

    argument_parser.add_argument("-if", "--input_folder",
                                 dest="input_folder",
                                 help="Please specify the path to the input folder",
                                 required=True)
    argument_parser.add_argument("-of", "--output_folder",
                                 dest="output_folder",
                                 help="Please specify the path to the output folder",
                                 required=True)
    argument_parser.add_argument("-nsol", "--solutions",
                                 dest="nsol",
                                 help="Please specify the number of solutions",
                                 required=True)
    argument_parser.add_argument("-he", "--heuristic",
                                 dest="heuristic",
                                 help="Please specify the heuristic",
                                 required=True)
    argument_parser.add_argument("-t", "--timeout",
                                 dest="timeout",
                                 help="Please specify the timeout",
                                 required=True)

    args = vars(argument_parser.parse_args())

    input_folder_path = args["input_folder"]
    output_folder_path = args["output_folder"]
    nsol = int(args["nsol"])
    timeout = int(args["timeout"])
    heuristic = args["heuristic"]

    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    files_list = os.listdir(input_folder_path)

    if output_folder_path[-1] != chr(92):
        output_folder_path += chr(92)
    if input_folder_path[-1] != chr(92):
        input_folder_path += chr(92)

    for (idx, file) in enumerate(files_list):
        file_out = "output" + str(idx + 1) + ".txt"
        i_file_path = input_folder_path + file
        o_file_path = output_folder_path + file_out

        output_file = open(o_file_path, "w")
        try:
            graph = Graph(i_file_path,output_file, timeout)
            breadth_first(graph, nsol)
            print("Done BFS")

            depth_first(graph, nsol)
            print("Done DFS")

            iterative_depth_first(graph, nsol)
            print("Done IDF")

            a_star(graph, nsol, heuristic=heuristic)
            print("Done A*")

            a_star_opt(graph, heuristic=heuristic,)
            print("Done A* opt")

            ida_star(graph, nsol, heuristic)
            print("Done IDA")

            graph.output_file.close()
        except Exception as e:
            print(str(e), file=output_file)
        output_file.close()

        print(f"Done for file {file}")



if __name__ == '__main__':
    argument_reader()
