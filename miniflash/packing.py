"""Cell packing: solver-free cell geometry from plane-level scheduling.

:func:`pack_cell` turns one region into a lattice-surgery cell without a SAT
solver, under the same contract the LaSsynth path obeys — K-in K-out at the
columns the floorplan assigned, in-pins at parity 0, out-pins carrying the exit
colour, ``pauli_frame`` from the region's own gates — so cells from either
engine mix freely in one layout.

Wire state. Every wire carries a physical colour ``c`` (the face table: on a K
wire of parity 0 the J faces are Z-type and the I faces X-type; parity 1 swaps
them) and a pending logical Hadamard ``b``. Wires enter at ``c = b = 0``: the
channel's flip landing has already paid whatever Hadamard the ledger owed. An
``h`` gate draws no geometry, it just flips ``b``. Every merge here is a J-face
merge, so what a gate needs is an exposure, and ``c ^ b`` is what decides it —
0 means the J faces are logically Z (a CX control, an S, a T injection), 1
means they are logically X (a CX target). Exit needs ``c == b``, the single bit
the wire ledger can carry.

``c ^ b`` is invariant under everything except the corkscrew: a K Hadamard wall
flips colour and pending together, while the corkscrew is a seven-turn detour
through the track and a neighbouring column that flips the colour alone with no
logical action. So a wire that changes role between gates must pay one, and no
routing choice avoids it — the dense band exposes only J faces, and the four
(colour, wall) forms of the CX staple all pin the colours to
``(b_control, 1 - b_target)``.

Geometry. Three tile rows: the wires at ``j = 2`` and a track above (jog row
``j = 4``, hops through 3) and below (``j = 0``, hops through 1). The unit of
time is a cube plane. A CX books two planes — the ancilla stretches along a
track carrying the control's merge on the first, lives one K unit of its own,
and takes the target on the second; that lifetime is why a CNOT cannot be one
plane — so a dependency chain costs one plane per level rather than a whole
slab. The corkscrew and the walled CX (the two wires arrive in the same colour,
so a Hadamard wall on the bridge separates the merges) span three planes; the S
staple spans two and a T injection one: the band's track rows are its
boundary, so the state is raised on the track tile in the qubit's own column
and the merge is the hop itself — the magic patch's temporal start is the open
end of that hop, and the Clifford skeleton of the gadget is a three-legged Z
spider that the factory's |T> closes. An operation claims a closed column
interval on one track at each plane it occupies, a wire merges at most once
per plane, and everything else on that wire is plain K pipe filled in
afterwards.

Scheduling is greedy by criticality (longest remaining chain, DasCot's reward):
the DAG frontier is walked in that order, a gate whose faces are exposed takes
the earliest plane with room, a gate whose wire is mis-exposed spends the plane
corkscrewing instead, and idle wires flip ahead towards their next role so the
flip hides behind other wires' gates instead of sitting on the critical path.
"""
from collections import deque

from .synthesis import PARITY_NONE, Cell, _assert_coords_in_bounds, _dual_stabilizers, _local_gates, _unsigned_stabilizers_and_fixup

WIRE_J = 2
HOP_J = 3
JOG_J = 4


def _pipe(i, j, k, axis, parity, hadamard=False):
    return {"kind": "hadamard" if hadamard else "pipe", "pos": [i, j, k], "axis": axis, "parity": parity}


def _corkscrew(base, colour, step, hop=HOP_J, jog=JOG_J):
    return 5, [
        _pipe(base, hop, 0, "J", colour),
        _pipe(base + step // 2, jog, 0, "I", colour),
        _pipe(base + step, jog, 1, "K", 1 - colour),
        _pipe(base + step // 2, jog, 2, "I", colour),
        _pipe(base, jog, 3, "K", 1 - colour),
        _pipe(base, hop, 4, "J", 1 - colour),
    ]


def _cx_blocks(control, target, colour_c, colour_t, walled, hop=HOP_J, jog=JOG_J):
    if not walled:
        blocks = [
            _pipe(control, WIRE_J, 1, "K", colour_c),
            _pipe(target, WIRE_J, 1, "K", colour_t),
            _pipe(control, hop, 0, "J", colour_c),
            _pipe(target, jog, 1, "K", 1 - colour_c),
            _pipe(target, hop, 2, "J", 1 - colour_c),
        ]
        step = 2 if target > control else -2
        for i in range(control, target, step):
            blocks.append(_pipe(i + step // 2, jog, 0, "I", colour_c))
        return 3, blocks

    blocks = [
        _pipe(control, WIRE_J, 1, "K", colour_c),
        _pipe(control, WIRE_J, 3, "K", colour_c),
        _pipe(target, WIRE_J, 1, "K", colour_t),
        _pipe(target, WIRE_J, 3, "K", colour_t),
        _pipe(control, hop, 2, "J", colour_c),
        _pipe(target, jog, 3, "K", 1 - colour_c, hadamard=True),
        _pipe(target, hop, 4, "J", colour_t),
    ]
    step = 2 if target > control else -2
    for i in range(control, target, step):
        blocks.append(_pipe(i + step // 2, jog, 2, "I", colour_c))
    return 5, blocks


def _t_blocks(column, colour, hop=HOP_J, jog=JOG_J):
    return 1, [_pipe(column, hop, 0, "J", colour)]


def _s_blocks(column, colour, hop=HOP_J, jog=JOG_J):
    return 3, [
        _pipe(column, WIRE_J, 1, "K", colour),
        {"kind": "ynode", "pos": [column, jog, 0], "parity": PARITY_NONE},
        _pipe(column, jog, 1, "K", colour),
        _pipe(column, hop, 2, "J", colour),
    ]


_TRACKS = ((HOP_J, JOG_J), (WIRE_J - 1, 0))


def _criticality(gates, qubits):
    crit = [0] * len(gates)
    last = {}
    for index in range(len(gates) - 1, -1, -1):
        name, gate_qubits = gates[index]
        weight = 0 if name == "h" else 1
        crit[index] = weight + max((crit[last[qubit]] for qubit in gate_qubits if qubit in last), default=0)
        for qubit in gate_qubits:
            last[qubit] = index
    return crit


def _role_want(name, gate_qubits, qubit):
    return 1 if name == "cx" and qubit == gate_qubits[1] else 0


class _Book:
    def __init__(self, tracks):
        self.tracks = tracks
        self.intervals = {}
        self.merging = {}

    def free(self, plane, track, low, high):
        return all(high < a or low > b for a, b in self.intervals.get((plane, track), ()))

    def take(self, plane, track, low, high):
        self.intervals.setdefault((plane, track), []).append((low, high))

    def idle(self, plane, qubits):
        return all(qubit not in self.merging.get(plane, ()) for qubit in qubits)

    def engage(self, plane, qubits):
        self.merging.setdefault(plane, set()).update(qubits)

    def place(self, spans, qubits_by_plane):
        """spans: [(plane, low, high)]; qubits_by_plane: {plane: qubits}. Returns a track or None."""
        for track in range(self.tracks):
            if all(self.free(plane, track, low, high) for plane, low, high in spans):
                for plane, low, high in spans:
                    self.take(plane, track, low, high)
                for plane, qubits in qubits_by_plane.items():
                    self.engage(plane, qubits)
                return track
        return None


def _schedule(region, gates, columns, width, tracks):
    crit = _criticality(gates, region.qubits)
    queue = {qubit: deque(index for index, (_, gate_qubits) in enumerate(gates) if qubit in gate_qubits) for qubit in region.qubits}
    colour = {qubit: 0 for qubit in region.qubits}
    pending = {qubit: 0 for qubit in region.qubits}
    book = _Book(tracks)
    ops = []
    landing = {}
    busy_until = {qubit: 0 for qubit in region.qubits}

    def drain_bookkeeping():
        for qubit in region.qubits:
            while queue[qubit] and gates[queue[qubit][0]][0] == "h":
                pending[qubit] ^= 1
                queue[qubit].popleft()

    def next_want(qubit):
        bit = pending[qubit]
        for index in queue[qubit]:
            name, gate_qubits = gates[index]
            if name == "h":
                bit ^= 1
                continue
            return bit ^ _role_want(name, gate_qubits, qubit)
        return bit

    def flip(plane, qubit):
        if busy_until[qubit] > plane:
            return False
        column = columns[qubit]
        for step in [s for s in (1, -1) if 1 <= column + s <= width - 2] or [1]:
            low, high = min(column, column + step), max(column, column + step)
            spans = [(plane + d, low, high) for d in (0, 1, 2)]
            if not book.idle(plane, (qubit,)) or not book.idle(plane + 1, (qubit,)) or not book.idle(plane + 2, (qubit,)):
                continue
            track = book.place(spans, {plane + d: (qubit,) for d in (0, 1, 2)})
            if track is None:
                continue
            ops.append({"kind": "flip", "plane": plane, "qubit": qubit, "step": step, "track": track, "colour": colour[qubit]})
            busy_until[qubit] = plane + 2
            landing.setdefault(plane + 2, []).append(qubit)
            return True
        return False

    drain_bookkeeping()
    plane = 0
    last = 0
    while any(queue.values()) or any(colour[qubit] != pending[qubit] for qubit in region.qubits):
        for qubit in landing.pop(plane, ()):
            colour[qubit] ^= 1
        frontier = sorted({queue[qubit][0] for qubit in region.qubits if queue[qubit] and all(queue[other] and queue[other][0] == queue[qubit][0] for other in gates[queue[qubit][0]][1])}, key=lambda index: (-crit[index], index))
        for index in frontier:
            name, gate_qubits = gates[index]
            if any(busy_until[qubit] > plane for qubit in gate_qubits):
                continue
            unready = [qubit for qubit in gate_qubits if colour[qubit] ^ pending[qubit] != _role_want(name, gate_qubits, qubit)]
            if unready:
                for qubit in unready:
                    flip(plane, qubit)
                continue
            if name == "cx":
                control, target = gate_qubits
                low, high = sorted((columns[control], columns[target]))
                walled = pending[control] != pending[target]
                if walled:
                    spans = [(plane, low, high), (plane + 1, low, high), (plane + 2, columns[target], columns[target])]
                    seats = {plane: (control,), plane + 2: (target,)}
                    reach = 2
                else:
                    spans = [(plane, low, high), (plane + 1, columns[target], columns[target])]
                    seats = {plane: (control,), plane + 1: (target,)}
                    reach = 1
                if not all(book.idle(p, q) for p, q in seats.items()):
                    continue
                track = book.place(spans, seats)
                if track is None:
                    continue
                ops.append({"kind": "cx", "plane": plane, "control": control, "target": target, "walled": walled,
                            "track": track, "colour_c": colour[control], "colour_t": colour[target]})
                for qubit in gate_qubits:
                    busy_until[qubit] = plane + reach
                last = max(last, plane + reach)
            elif name in ("t", "tdg"):
                qubit = gate_qubits[0]
                column = columns[qubit]
                if not book.idle(plane, (qubit,)):
                    continue
                track = book.place([(plane, column, column)], {plane: (qubit,)})
                if track is None:
                    continue
                ops.append({"kind": "t", "plane": plane, "qubit": qubit, "dagger": name == "tdg",
                            "track": track, "colour": colour[qubit]})
                busy_until[qubit] = plane
                last = max(last, plane)
            else:
                qubit = gate_qubits[0]
                column = columns[qubit]
                spans = [(plane, column, column), (plane + 1, column, column)]
                if not book.idle(plane + 1, (qubit,)):
                    continue
                track = book.place(spans, {plane + 1: (qubit,)})
                if track is None:
                    continue
                ops.append({"kind": "s", "plane": plane, "qubit": qubit, "track": track, "colour": colour[qubit]})
                busy_until[qubit] = plane + 1
                last = max(last, plane + 1)
            for qubit in gate_qubits:
                queue[qubit].popleft()
        for qubit in sorted(region.qubits, key=lambda qubit: (len(queue[qubit]) == 0, -crit[queue[qubit][0]] if queue[qubit] else 0)):
            if busy_until[qubit] <= plane and colour[qubit] != next_want(qubit) and flip(plane, qubit):
                last = max(last, plane + 2)
        drain_bookkeeping()
        plane += 1
        if plane - (ops[-1]["plane"] if ops else 0) > 24:
            raise RuntimeError("pack_cell: no progress — every track blocked")
    for qubit, landed in landing.items():
        last = max(last, qubit)
    return ops, colour, last


def pack_cell(region, in_columns=None, in_bases=None, tracks=None) -> Cell:
    """Pack a region's gates into slabs on one or two tracks.

    :param region: Region.
    :param in_columns: {qubit: local column}; kept at entry and exit.
    :param in_bases: {qubit: 0|1} wire ledger bits at entry, recorded on the
        cell; the geometry starts every wire at colour 0 with nothing pending.
    :param tracks: 1 (jog row above only), 2 (both) or None to pack both
        ways and keep the smaller footprint, counting the extra tile row
        the lower track costs a single-row layout.
    :returns: Cell.
    :raises ValueError: on a gate with no pattern.
    """
    columns = dict(in_columns) if in_columns is not None else {qubit: index + 1 for index, qubit in enumerate(region.qubits)}
    width = max(len(region.qubits) + 2, max(columns.values()) + 2)
    gates = [(name, tuple(gate_qubits)) for name, gate_qubits in region.gates if name != "x"]
    for name, _ in gates:
        if name not in ("h", "s", "sdg", "cx", "t", "tdg"):
            raise ValueError(f"pack_cell: unsupported gate {name!r}")
    if tracks is None:
        double = _schedule(region, gates, columns, width, 2)
        try:
            single = _schedule(region, gates, columns, width, 1)
        except RuntimeError:
            single = None
        ops, colour, last = single if single is not None and single[2] * 2 <= double[2] * 3 else double
    else:
        ops, colour, last = _schedule(region, gates, columns, width, tracks)

    blocks = []
    injections = []
    covered = {qubit: set() for qubit in region.qubits}
    vacated = {qubit: set() for qubit in region.qubits}
    timeline = {qubit: [(0, 0)] for qubit in region.qubits}

    for op in sorted(ops, key=lambda op: op["plane"]):
        base = 2 * op["plane"]
        hop, jog = _TRACKS[op["track"]]
        if op["kind"] == "flip":
            qubit = op["qubit"]
            _, stage = _corkscrew(2 * columns[qubit], op["colour"], 2 * op["step"], hop, jog)
            vacated[qubit].update((op["plane"], op["plane"] + 1))
            timeline[qubit].append((op["plane"] + 2, 1 - op["colour"]))
        elif op["kind"] == "cx":
            control, target = op["control"], op["target"]
            _, stage = _cx_blocks(2 * columns[control], 2 * columns[target], op["colour_c"], op["colour_t"], op["walled"], hop, jog)
            slots = (op["plane"], op["plane"] + 1) if op["walled"] else (op["plane"],)
            covered[control].update(slots)
            covered[target].update(slots)
        elif op["kind"] == "t":
            qubit = op["qubit"]
            _, stage = _t_blocks(2 * columns[qubit], op["colour"], hop, jog)
            injections.append({"qubit": qubit, "dagger": op["dagger"], "basis": op["colour"],
                               "offset": [2 * columns[qubit], jog, base]})
        else:
            qubit = op["qubit"]
            _, stage = _s_blocks(2 * columns[qubit], op["colour"], hop, jog)
            covered[qubit].add(op["plane"])
        for block in stage:
            position = list(block["pos"])
            position[2] += base
            blocks.append({**block, "pos": position})

    def colour_in(qubit, slot):
        value = 0
        for start_slot, shade in timeline[qubit]:
            if start_slot <= slot:
                value = shade
        return value

    for qubit in region.qubits:
        for slot in range(last):
            if slot in covered[qubit] or slot in vacated[qubit]:
                continue
            blocks.append(_pipe(2 * columns[qubit], WIRE_J, 2 * slot + 1, "K", colour_in(qubit, slot)))
    height = 2 * last + 1

    pins = [{"name": region.in_pins[qubit], "dir": "K", "end": "-", "parity": 0, "offset": [2 * columns[qubit], WIRE_J, -1]} for qubit in region.qubits]
    pins += [{"name": region.out_pins[qubit], "dir": "K", "end": "+", "parity": colour[qubit], "offset": [2 * columns[qubit], WIRE_J, height]} for qubit in region.qubits]

    seen = set()
    for block in blocks:
        key = tuple(block["pos"])
        if key in seen:
            raise RuntimeError(f"pack_cell: block collision at {key}")
        seen.add(key)
    dims = [2 * width - 1, 5, height]
    _assert_coords_in_bounds(blocks, dims)
    clifford = [(name, qubits) for name, qubits in _local_gates(region) if name not in ("t", "tdg")]
    _, frame = _unsigned_stabilizers_and_fixup(_dual_stabilizers(clifford, len(region.qubits)))
    return Cell(dims=dims, pins=pins, blocks=blocks, pauli_frame=frame, in_bases={qubit: (in_bases or {}).get(qubit, 0) for qubit in region.qubits}, injections=injections)
