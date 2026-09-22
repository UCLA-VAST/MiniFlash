"""The 1-D floorplan solver.

Column-stable placement: every qubit keeps a home column for the whole
layout while cell boxes float around them. Between layers, wires the
next cell does not consume are swapped out to straight parking lanes and
swapped back on demand; the resulting channel rearrangements are solved
by :func:`miniflash.channel.solve`. A pure function; there is no search.
"""
from .floorplan import Floorplan


def _stable_layer_columns(layers, num_qubits):
    num_layers = len(layers)
    position = {}
    ever_occupied = set()
    backfill = []
    in_port_columns, out_port_columns, lane_columns, box_layouts = [], [], [], []

    for layer in range(num_layers):
        members = {qubit for box in layers[layer] for qubit in box}
        parked = [qubit for qubit in range(num_qubits) if qubit not in members]
        parked_cols = {position[qubit] for qubit in parked if qubit in position}
        boxes = layers[layer]
        if len(boxes) > 1:
            boxes = sorted(boxes, key=lambda box: sum(position.get(qubit, num_qubits + qubit) for qubit in box) / len(box))

        layout, in_cols, out_cols = [], {}, {}
        placed = set()
        for box in boxes:
            seen = sorted((qubit for qubit in box if qubit in position), key=lambda qubit: position[qubit])
            unseen = sorted(qubit for qubit in box if qubit not in position)
            ordered = seen + unseen
            width = len(box) + 2
            ideal = max(0, position[seen[len(seen) // 2]] - len(seen) // 2 - 1) if seen else min(ever_occupied ^ ever_occupied | {0}) if not ever_occupied else max(ever_occupied) + 1
            blocked = parked_cols | placed

            best = None
            limit = max(ever_occupied, default=0) + width + num_qubits + 2
            for offset in range(0, limit):
                window = range(offset, offset + width)
                if any(column in blocked for column in window):
                    continue
                if any(offset + 1 + rank in ever_occupied for rank in range(len(seen), len(ordered))):
                    continue
                score = sum(1 for rank, qubit in enumerate(seen) if offset + 1 + rank != position[qubit])
                key = (score, abs(offset - ideal))
                if best is None or key < best[0]:
                    best = (key, offset)

            offset = best[1]
            placed.update(range(offset, offset + width))
            ever_occupied.update(range(offset, offset + width))
            for rank, qubit in enumerate(ordered):
                in_cols[qubit] = offset + rank + 1
                out_cols[qubit] = offset + rank + 1
            for qubit in unseen:
                backfill.append((qubit, in_cols[qubit], layer))
            layout.append((0, offset, list(box)))

        lane_columns.append({qubit: position[qubit] for qubit in parked if qubit in position})
        in_port_columns.append(in_cols)
        out_port_columns.append(out_cols)
        box_layouts.append(layout)
        for qubit in members:
            position[qubit] = out_cols[qubit]
            ever_occupied.add(out_cols[qubit])

    cursor = 0
    for qubit in range(num_qubits):
        if qubit not in position:
            while cursor in ever_occupied:
                cursor += 1
            backfill.append((qubit, cursor, num_layers))
            ever_occupied.add(cursor)
            cursor += 1

    for qubit, column, first_layer in backfill:
        for layer in range(first_layer):
            lane_columns[layer][qubit] = column
        for layer in range(first_layer, num_layers):
            if qubit not in in_port_columns[layer] and qubit not in lane_columns[layer]:
                lane_columns[layer][qubit] = column

    return in_port_columns, out_port_columns, lane_columns, box_layouts




def solve_1d(layers, num_qubits):
    """Solve the 1-D floorplan: port columns, lanes, channel moves, gap levels.

    :param layers: list per layer of per-box qubit lists.
    :param num_qubits: int.
    :returns: Floorplan.
    :raises ValueError: on empty/duplicate layers.
    """
    for layer, boxes in enumerate(layers):
        union = [qubit for box in boxes for qubit in box]
        if not union:
            raise ValueError(f"solve_floorplan: layer {layer} has no members")
        if not set(union) <= set(range(num_qubits)):
            raise ValueError(f"solve_floorplan: layer {layer} members out of range")
        if len(set(union)) != len(union):
            raise ValueError(f"solve_floorplan: layer {layer} has duplicate members")

    num_layers = len(layers)
    box_widths = [sum(len(box) + 2 for box in boxes) for boxes in layers]
    max_box_width = max(box_widths)
    num_lanes = 0
    in_port_columns, out_port_columns, lane_columns, box_layouts = _stable_layer_columns(layers, num_qubits)
    max_box_width = max((max((offset + len(box) + 2 for _, offset, box in layout), default=0) for layout in box_layouts), default=max_box_width)
    max_box_width = max(max_box_width, max((col + 1 for lanes in lane_columns for col in lanes.values()), default=0))
    first_lane_column = max_box_width

    scratch_column = first_lane_column + num_lanes
    any_cycle = False
    moves, gap_levels = [], []
    for channel in range(num_layers - 1):
        before = dict(out_port_columns[channel])
        before.update(lane_columns[channel])
        after = dict(in_port_columns[channel + 1])
        after.update(lane_columns[channel + 1])

        from .channel import Lane, RearrangementChannel, solve

        lanes = {qubit: Lane(start=(0, before[qubit]), end=(0, after[qubit])) for qubit in sorted(before) if qubit in after}
        plan = solve(RearrangementChannel(lanes=lanes, scratch=((0, scratch_column),)))
        if len(plan.moves) > sum(1 for lane in lanes.values() if lane.start != lane.end):
            any_cycle = True
        moves.append(list(plan.moves))
        gap_levels.append(plan.levels)

    return Floorplan(num_qubits=num_qubits, box_widths=box_widths, in_port_columns=in_port_columns, out_port_columns=out_port_columns, lane_columns=lane_columns, moves=moves, gap_levels=gap_levels, placements=box_layouts)
