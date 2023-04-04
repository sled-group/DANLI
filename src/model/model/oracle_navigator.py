import math
from typing import Dict, List, Tuple, Set
from collections import deque


def distance(loc1, loc2):
    if isinstance(loc1, dict):
        return ((loc1["x"] - loc2["x"]) ** 2 + (loc1["z"] - loc2["z"]) ** 2) ** 0.5
    elif isinstance(loc1, tuple):
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[2] - loc2[2]) ** 2) ** 0.5
    else:
        raise TypeError(
            "Location must be a dict with keys 'x', 'y' and 'z' or a 3-element tuple."
        )


def closest_position(
    object_position: Dict[str, float], reachable_positions: List[Dict[str, float]]
) -> Dict[str, float]:
    out = reachable_positions[0]
    min_distance = float("inf")
    for pos in reachable_positions:
        # NOTE: y is the vertical direction, so only care about the x/z ground positions
        dist = distance(pos, object_position)
        if dist < min_distance:
            min_distance = dist
            out = pos
    return out


def closest_grid_point(
    point: Tuple[float, float, float], grid_size=0.25
) -> Tuple[float, float, float]:
    """Return the grid point that is closest to a world coordinate.

    Expects world_point=(x_pos, y_pos, z_pos). Note y_pos is ignored in the calculation.
    """
    gs = grid_size
    return (round(point[0] / gs) * gs, point[1], round(point[2] / gs) * gs)


def get_navi_grid_neighbors(reachable_positions_tuple, grid_size=0.25):
    """Return a dict for the neighbor positions of all reachable positions"""
    neighbors = dict()
    for position in reachable_positions_tuple:
        position_neighbors = set()
        for p in reachable_positions_tuple:
            if position != p and (
                (
                    abs(position[0] - p[0]) < 1.5 * grid_size
                    and abs(position[2] - p[2]) < 0.5 * grid_size
                )
                or (
                    abs(position[0] - p[0]) < 0.5 * grid_size
                    and abs(position[2] - p[2]) < 1.5 * grid_size
                )
            ):
                position_neighbors.add(p)
        neighbors[position] = position_neighbors
        # {k: len(v) for k, v in neighbors.items()}

    return neighbors


def shortest_path(
    navigation_graph: Dict[Tuple[float, float, float], Set[Tuple[float, float, float]]],
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    grid_size=0.25,
):
    """Expects the start=(x_pos, y_pos, z_pos) and end=(x_pos, y_pos, z_pos).

    Note y_pos is ignored in the calculation.
    """

    if start == end:
        return [start]

    q = deque()
    q.append([start])

    visited = set()

    while q:
        path = q.popleft()
        pos = path[-1]

        if pos in visited:
            continue

        visited.add(pos)
        for neighbor in navigation_graph[pos]:
            if neighbor == end:
                return path + [neighbor]
            q.append(path + [neighbor])

    raise Exception("Invalid state. Must be a bug!")


class OracleNavigator:
    def __init__(self, logger):
        """
        transformer model of high-level planning
        """
        self.grid_size = 0.25
        self.head_above_agent_y = 0.1
        self.logger = logger

        # plot_shortest_path(reachable_positions_tuple, neighbors, start, end)
    def reset(self, env):
        self.env = env
        # get reachable positions in the scene
        self.reachable_dict = env.step(action="GetReachablePositions").metadata[
            "actionReturn"
        ]
        self.reachable_tuple = [(p["x"], p["y"], p["z"]) for p in self.reachable_dict]
        self.navigation_graph = get_navi_grid_neighbors(self.reachable_tuple)


    def step(self, target_object_types):
        """
        return single-step predictions during inference
        """
        env = self.env

        tobj = target_object_types

        agent_position = env.last_event.metadata["agent"]["position"]
        agent_rotation = env.last_event.metadata["agent"]["rotation"]["y"]
        agent_horizon = env.last_event.metadata["agent"]["cameraHorizon"]

        tobj_meta, closest_distance = None, 100000
        for o in env.last_event.metadata["objects"]:
            if o["objectType"] in tobj:
                dist = distance(o["position"], agent_position)
                if o["visible"]:
                    dist -= 0.5
                if dist < closest_distance:
                    closest_distance = dist
                    tobj_meta = o
        if not tobj_meta:
            self.logger.info("Object [%s] not exists in the scene" % tobj)
            return "Stop"

        print(tobj_meta["objectType"], tobj_meta["position"])
        print("agent_position", agent_position)
        # from pprint import pprint
        # pprint(tobj_meta)
        tobj_position = tobj_meta["position"]

        print('   closest_distance:', closest_distance)

        # get the closest interaction position to the target object
        closest = closest_position(tobj_meta["position"], self.reachable_dict)
        start = (agent_position["x"], agent_position["y"], agent_position["z"])
        start = closest_grid_point(start, self.grid_size)
        end = (closest["x"], closest["y"], closest["z"])
        end = closest_grid_point(end, self.grid_size)

        if start == end:  # closest_distance < 1.5:
            # close enough to the target object
            print('   close enough to the target object')

            # check if the object is visible
            # visible = tobj_meta["visible"]
            # if visible:
            #     # oid = tobj_meta["objectId"]
            #     # bbox = env.last_event.instance_detections2D[oid]
            #     # center = [
            #     #     (bbox[0] + bbox[2]) / (2 * 300),
            #     #     (bbox[1] + bbox[3]) / (2 * 300),
            #     # ]
            #     # if center[1] > 0.15 and center[1] < 0.85:
            #     #     # print('   object in the agent central view: done!')
            #     return "Stop"

            print('   object not visible: try to rotate agent to look at it')
            # object not visible: try to rotate agent to look at it
            diff_x = tobj_position["x"] - agent_position["x"]
            diff_y = tobj_position["y"] - agent_position["y"]
            diff_z = tobj_position["z"] - agent_position["z"]

            rot_xz = math.atan(diff_z / (diff_x + 1e-8)) / math.pi * 180

            if abs(rot_xz) >= 45:  # north/south: z axis
                diff_xz = diff_z
                if diff_z >= 0:
                    abs_rotation = 0
                else:
                    abs_rotation = 180

            else:  # east/west
                diff_xz = diff_x
                if diff_x >= 0:  # east
                    abs_rotation = 90
                else:  # west
                    abs_rotation = 270

            diff_rotation = abs_rotation - round(agent_rotation)
            if diff_rotation < 0:
                diff_rotation += 360

            # print('      ', round(agent_rotation), abs_rotation)
            # print('      ', diff_rotation)

            if diff_rotation == 270 or diff_rotation == 180:
                return "Turn Left"
            elif diff_rotation == 90:
                return "Turn Right"

            # print('   rotation is good: finally tune the agent camera pitch')
            # rotation is good: finally tune the agent camera's pitch
            diff_y -= self.head_above_agent_y
            rot_y = math.atan(diff_y / (diff_xz + 1e-8)) / math.pi * 180

            # if diff_y > 0:  # the object is above the agent
            #     return 'Look Up'
            # else:  # the object is below the agent
            #     return 'Look Down'

            # print('      ', round(agent_horizon), diff_y, rot_y)

            if diff_y > 0:  # the object is above the agent
                if abs(rot_y) > 45:
                    if round(agent_horizon) >= 0:
                        return "Look Up"
                    else:
                        pass
                else:
                    if round(agent_horizon) >= 30:
                        return "Look Up"
                    else:
                        pass
            else:  # the object is below the agent
                if abs(rot_y) < 30:
                    if round(agent_horizon) <= 0:
                        return "Look Down"
                    else:
                        pass
                else:
                    if round(agent_horizon) <= 30:
                        return "Look Down"
                    else:
                        pass
            if tobj_meta["visible"]:
                print('   object is visible: done!')
                return "Stop"

            print('   horizon is good but the object still invisible: the object may inside some closed container')
            # horizon is good: the object may inside some closed container
            return "Stop"

        else:
            # not get close enough to the target object: evoke path planning
            print('   not get close enough to the target object: evoke path planning')

            if start == end:
                # the closest point is still not enough for interaction
                # end navigation
                return "Stop"

            plan = shortest_path(self.navigation_graph, start, end)
            next_node = plan[1]
            print('      ', start, next_node)

            diff_x = round((next_node[0] - start[0]) * 100)
            diff_z = round((next_node[2] - start[2]) * 100)

            mapping = {
                (0, 0, 25): "Forward",
                (0, 0, -25): "Turn Right",
                (0, 25, 0): "Turn Right",
                (0, -25, 0): "Turn Left",
                
                (180, 0, 25): "Turn Right",
                (180, 0, -25): "Forward",
                (180, 25, 0): "Turn Left",
                (180, -25, 0): "Turn Right",
                
                (90, 0, 25): "Turn Left",
                (90, 0, -25): "Turn Right",
                (90, 25, 0): "Forward",
                (90, -25, 0): "Turn Right",
                
                (270, 0, 25): "Turn Right",
                (270, 0, -25): "Turn Left",
                (270, 25, 0): "Turn Right",
                (270, -25, 0): "Forward",
            }

            situation = (round(agent_rotation), diff_x, diff_z)
            print('      ', situation)

            return mapping[situation]
