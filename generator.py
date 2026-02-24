#!/usr/bin/env python3
"""
Laser Cut Sheet Optimizer — ILP/FFD 2-D Bin Packing + SVG Output
=================================================================
Generates ready-to-use SVG files for laser engraving/cutting machines.

Architecture
------------
  parse_csv()          Read label definitions from a semicolon-delimited CSV.
  ILPBinPacker         Optimal 2-D bin packing via OR-Tools CP-SAT (default).
  SimpleBinPacker      Fast FFD heuristic fallback (no external dependencies).
  SVGGenerator         Render a packed Sheet to an SVG with three named layers
                         (back to front):
                         cut_lines — red 0.1 mm cutting outline
                         borders   — amber 0.8 mm etched frame around each label
                         text      — blue text converted to merged path outlines

Usage
-----
    python generator.py INPUT.csv 300 200
    python generator.py INPUT.csv 800 600 -o my_project
    python generator.py INPUT.csv 300 200 --solver ffd
    python generator.py INPUT.csv 300 200 --time-limit 120

Input CSV Format
----------------
    "QUANTITY";"MATERIAL";"TEXTDATA";"LABEL WIDTH(mm)";"LABEL HEIGHT(mm)";"TEXT HEIGHT(mm)"
    Text fields support embedded newlines (\\n) for multi-line labels.

Output SVG Colour Convention
-----------------------------
    #f44336  (red)   — laser cutting path
    #ffc107  (amber) — laser engraving fill  (0.8 mm border frame)
    #0000ff  (blue)  — label text rendered as filled path outlines

Dependencies
------------
    Required : svgwrite, fonttools
    Optional : ortools  (ILP solver; falls back to FFD if absent)
"""

import csv
import math
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Label:
    """
    Immutable specification for one label type.

    Attributes
    ----------
    material    : Stock material identifier (used to group labels onto sheets).
    width       : Nominal label width in millimetres.
    height      : Nominal label height in millimetres.
    text        : Engraved text.  Newlines (\\n) separate visual lines.
    text_height : Cap-height of the largest text line in millimetres.
    quantity    : Number of identical copies required.
    """

    material: str
    width: float        # mm
    height: float       # mm
    text: str
    text_height: float  # mm
    quantity: int = 1

    @property
    def area(self) -> float:
        """Bounding-box area in mm² (width × height, ignoring rotation)."""
        return self.width * self.height


@dataclass
class Placement:
    """
    Position of a single label instance on a sheet.

    Attributes
    ----------
    label    : The label specification being placed.
    x        : Left-edge X coordinate in mm (sheet origin at top-left).
    y        : Top-edge Y coordinate in mm.
    rotation : 0 = natural orientation; 90 = label rotated 90° clockwise.
    """

    label: Label
    x: float
    y: float
    rotation: int = 0  # 0 or 90 degrees


@dataclass
class Sheet:
    """
    One physical stock sheet and the label placements it carries.

    Attributes
    ----------
    width      : Sheet width in mm.
    height     : Sheet height in mm.
    placements : Ordered list of all label placements on this sheet.
    material   : Stock material identifier (matches Label.material).
    """

    width: float
    height: float
    placements: List[Placement]
    material: str

    @property
    def efficiency(self) -> float:
        """
        Material utilisation as a percentage [0, 100].

        Calculated as the sum of all placed label bounding-box areas
        divided by the total sheet area.  Does not account for kerf width
        or border etching margins.
        """
        used_area  = sum(p.label.area for p in self.placements)
        total_area = self.width * self.height
        return (used_area / total_area * 100) if total_area > 0 else 0


# ============================================================================
# SIMPLE BIN PACKER FOR 2D BIN PACKING  (used as FFD fallback / warm-start)
# ============================================================================

class SimpleBinPacker:
    """
    First-Fit Decreasing (FFD) heuristic for 2-D bin packing.

    Algorithm overview
    ------------------
    1. Sort all items by bounding-box area, largest first (FFD ordering).
    2. For each item, try both 0° and 90° orientations.
    3. Use a skyline strategy to determine the lowest feasible Y position
       at each candidate X coordinate on the current sheet.
    4. Open a new sheet whenever the current one cannot accommodate the item.

    This class is used both as a standalone fast solver (--solver ffd) and
    as a warm-start heuristic for ILPBinPacker.

    No external dependencies required.
    """

    def __init__(self, sheet_width: float, sheet_height: float) -> None:
        """
        Initialise the packer with fixed sheet dimensions.

        Parameters
        ----------
        sheet_width  : Available sheet width in mm.
        sheet_height : Available sheet height in mm.
        """
        self.sheet_width  = sheet_width
        self.sheet_height = sheet_height
        
    def pack(self, labels: List[Label]) -> List[Sheet]:
        """
        Pack all label instances onto the minimum number of sheets.

        Labels are first grouped by material so that each output sheet
        contains only one material type.  Within each group the FFD
        ordering (largest area first) is applied before placement.

        Parameters
        ----------
        labels : Label specifications including their quantities.

        Returns
        -------
        List of Sheet objects, one per required physical sheet.
        """
        # Expand each label by its quantity so every instance is a
        # separate entry in the per-material queues.
        by_material: Dict[str, List[Label]] = defaultdict(list)
        for label in labels:
            for _ in range(label.quantity):
                by_material[label.material].append(label)
        
        all_sheets = []
        
        for material, material_labels in by_material.items():
            # Sort by area (largest first) - FFD heuristic
            sorted_labels = sorted(material_labels, 
                                  key=lambda l: l.area, 
                                  reverse=True)
            
            sheets = self._pack_material(sorted_labels, material)
            all_sheets.extend(sheets)
        
        return all_sheets
    
    def _pack_material(self, labels: List[Label], material: str) -> List[Sheet]:
        """
        Pack a homogeneous list of label instances (same material) onto sheets.

        Opens new sheets on demand whenever the current one is full.  Labels
        are expected to be pre-sorted by area (largest first) before this
        method is called.

        Parameters
        ----------
        labels   : Pre-sorted list of label instances to place.
        material : Material identifier applied to every created Sheet.

        Returns
        -------
        List of Sheet objects for this material group.
        """
        sheets: List[Sheet] = []
        current_sheet: Optional[Sheet] = None
        
        for label in labels:
            # Try to fit in existing sheet
            placed = False
            
            if current_sheet:
                placement = self._find_placement(current_sheet, label)
                if placement:
                    current_sheet.placements.append(placement)
                    placed = True
            
            # Create new sheet if needed
            if not placed:
                current_sheet = Sheet(
                    width=self.sheet_width,
                    height=self.sheet_height,
                    placements=[],
                    material=material
                )
                sheets.append(current_sheet)
                
                # Place in new sheet
                placement = self._find_placement(current_sheet, label)
                if placement:
                    current_sheet.placements.append(placement)
                else:
                    print(f"Warning: Label {label.width}x{label.height} "
                          f"doesn't fit in sheet {self.sheet_width}x{self.sheet_height}")
        
        return sheets
    
    def _find_placement(self, sheet: Sheet, label: Label) -> Optional[Placement]:
        """
        Find a valid placement for *label* on *sheet*, or return None.

        Both 0° and 90° orientations are tried in preference order.
        The first orientation that fits (using the skyline strategy) is
        returned without exploring further alternatives.

        Parameters
        ----------
        sheet : Target sheet with existing placements.
        label : Label to be placed.

        Returns
        -------
        Placement if a feasible position was found, else None.
        """
        
        # Try both orientations
        orientations = [
            (label.width, label.height, 0),
            (label.height, label.width, 90)
        ]
        
        for width, height, rotation in orientations:
            # Check if fits in sheet
            if width > sheet.width or height > sheet.height:
                continue
            
            # Find free space using skyline algorithm
            placement = self._skyline_place(sheet, width, height, label, rotation)
            if placement:
                return placement
        
        return None
    
    def _skyline_place(self, sheet: Sheet, width: float, height: float,
                       label: Label, rotation: int) -> Optional[Placement]:
        """
        Attempt to place a rectangle of given *width* × *height* on *sheet*.

        Iterates over each X-column defined by the current skyline.  For
        each candidate X the minimum Y that avoids all existing placements
        is computed; if the rectangle fits within the sheet bounds and
        passes an overlap check, a Placement is returned.

        Parameters
        ----------
        sheet    : Target sheet.
        width    : Effective width of the item in chosen orientation (mm).
        height   : Effective height of the item in chosen orientation (mm).
        label    : Label specification (stored in the returned Placement).
        rotation : 0 or 90 — the chosen orientation.

        Returns
        -------
        Placement on success, None if no feasible position exists.
        """
        
        # Build skyline from existing placements
        skyline = self._build_skyline(sheet)
        
        # Try to place at each skyline position
        for x, y_base in skyline:
            # Check if fits
            if x + width > sheet.width:
                continue
            
            # Check vertical space
            max_y = self._get_max_y_at_position(sheet, x, width)
            y = max(y_base, max_y)
            
            if y + height <= sheet.height:
                # Check for overlaps
                if not self._check_overlap(sheet, x, y, width, height):
                    return Placement(label, x, y, rotation)
        
        return None
    
    def _build_skyline(self, sheet: Sheet) -> List[Tuple[float, float]]:
        """
        Build a simplified skyline from the current placements on *sheet*.

        The skyline is represented as a sorted list of (x, y) knot points
        where x is the right edge of each existing placement and y is
        always 0 (the sheet top).  This is a coarse approximation that
        lists column starts rather than a true stepped profile.

        Parameters
        ----------
        sheet : Sheet whose placements define the skyline.

        Returns
        -------
        Sorted, deduplicated list of (x, 0) knot points.  Returns
        [(0, 0)] when the sheet is empty.
        """
        if not sheet.placements:
            return [(0, 0)]
        
        # Simple approach: check grid positions
        skyline = [(0, 0)]
        
        # Add positions after each placement
        for p in sheet.placements:
            w = p.label.width if p.rotation == 0 else p.label.height
            skyline.append((p.x + w, 0))
        
        # Sort and deduplicate
        skyline = sorted(set(skyline))
        return skyline
    
    def _get_max_y_at_position(self, sheet: Sheet, x: float, width: float) -> float:
        """
        Return the minimum Y that avoids all existing placements in an X band.

        Scans every placed label whose X extent overlaps [x, x+width] and
        returns the maximum bottom edge among those labels.  This value is
        the earliest Y at which a new item can start without vertical overlap.

        Parameters
        ----------
        sheet : Sheet with existing placements.
        x     : Left edge of the candidate X band (mm).
        width : Width of the candidate item (mm).

        Returns
        -------
        Minimum safe Y coordinate (mm); 0.0 if the band is empty.
        """
        max_y = 0
        
        for p in sheet.placements:
            pw = p.label.width if p.rotation == 0 else p.label.height
            ph = p.label.height if p.rotation == 0 else p.label.width
            
            # Check if this placement overlaps X range
            if not (p.x + pw <= x or p.x >= x + width):
                max_y = max(max_y, p.y + ph)
        
        return max_y
    
    def _check_overlap(self, sheet: Sheet, x: float, y: float,
                       width: float, height: float) -> bool:
        """
        Return True if the candidate rectangle overlaps any existing placement.

        Uses axis-aligned bounding-box intersection: two rectangles are
        considered *non*-overlapping when one is entirely to the left, right,
        above, or below the other.

        Parameters
        ----------
        sheet  : Sheet with existing placements.
        x      : Candidate left edge (mm).
        y      : Candidate top edge (mm).
        width  : Candidate width (mm).
        height : Candidate height (mm).

        Returns
        -------
        True if any collision is detected, False otherwise.
        """
        
        for p in sheet.placements:
            pw = p.label.width if p.rotation == 0 else p.label.height
            ph = p.label.height if p.rotation == 0 else p.label.width
            
            # Check rectangle intersection
            if not (x + width <= p.x or x >= p.x + pw or
                   y + height <= p.y or y >= p.y + ph):
                return True
        
        return False


# ============================================================================
# ILP BIN PACKER  (OR-Tools CP-SAT)
# ============================================================================

class ILPBinPacker:
    """
    Optimal 2D Bin Packing via Integer Linear Programming (OR-Tools CP-SAT).

    Formulation
    -----------
    For each item i (label instance), sheet k, and orientation o ∈ {0°, 90°}:

        a[i,k,o]  BoolVar  — item i placed on sheet k with orientation o
        x[i,k,o]  IntVar   — left edge position (integer, scaled by SCALE)
        y[i,k,o]  IntVar   — top  edge position (integer, scaled by SCALE)

    Constraints
    -----------
    1. Each item placed exactly once:
           ∑_k ∑_o  a[i,k,o] = 1   ∀ i
    2. No overlap on each sheet (AddNoOverlap2D with optional intervals)
    3. Sheet occupancy indicator:
           sheet_used[k] ≥ a[i,k,o]  ∀ i, o
    4. Symmetry breaking (used sheets are contiguous):
           sheet_used[k] ≥ sheet_used[k+1]

    Objective:  minimise  ∑_k  sheet_used[k]

    The FFD heuristic solution is used as:
      - Upper bound → limits the number of candidate sheets
      - Hint        → warm-starts the CP-SAT search
    """

    SCALE = 10  # 0.1 mm integer precision

    def __init__(self, sheet_width: float, sheet_height: float,
                 time_limit: int = 60) -> None:
        """
        Initialise the ILP packer.

        Parameters
        ----------
        sheet_width  : Available sheet width in mm.
        sheet_height : Available sheet height in mm.
        time_limit   : Maximum CP-SAT wall-clock time in seconds per material
                       group.  The solver returns the best feasible solution
                       found within this budget; the FFD result is used as a
                       fall-back if CP-SAT finds nothing better.
        """
        self.sheet_width  = sheet_width
        self.sheet_height = sheet_height
        self.time_limit   = time_limit

    def pack(self, labels: List[Label]) -> List[Sheet]:
        """
        Pack all label instances onto the minimum number of sheets.

        Groups labels by material, sorts each group by area (largest first)
        for a tighter ILP search space, then delegates to _pack_material.

        Parameters
        ----------
        labels : Label specifications including their quantities.

        Returns
        -------
        List of Sheet objects — each sheet contains only one material type.
        """
        by_material: Dict[str, List[Label]] = defaultdict(list)
        for label in labels:
            for _ in range(label.quantity):
                by_material[label.material].append(label)

        all_sheets: List[Sheet] = []
        for material, mat_labels in by_material.items():
            # Largest-area-first ordering as in FFD
            sorted_labels = sorted(mat_labels, key=lambda l: l.area, reverse=True)
            all_sheets.extend(self._pack_material(sorted_labels, material))
        return all_sheets

    # ------------------------------------------------------------------
    # Per-material dispatch
    # ------------------------------------------------------------------

    def _pack_material(self, labels: List[Label], material: str) -> List[Sheet]:
        """
        Dispatch packing for a single material group.

        Attempts to import OR-Tools.  Falls back to SimpleBinPacker (FFD)
        if ortools is not available, printing a notice to stdout.

        Parameters
        ----------
        labels   : Label instances pre-sorted by area (largest first).
        material : Material identifier applied to every created Sheet.

        Returns
        -------
        List of Sheet objects produced by whichever solver is available.
        """
        try:
            import ortools  # noqa: F401 — presence check only
        except ImportError:
            print("  ⚠️  OR-Tools not installed.  pip install ortools")
            print("  ↩  Falling back to FFD heuristic")
            return SimpleBinPacker(self.sheet_width, self.sheet_height) \
                       ._pack_material(labels, material)
        return self._solve_cp_sat(labels, material)

    # ------------------------------------------------------------------
    # CP-SAT solver
    # ------------------------------------------------------------------

    def _solve_cp_sat(self, labels: List[Label], material: str) -> List[Sheet]:
        """
        Run the CP-SAT model and return packed sheets for one material group.

        Steps
        -----
        1. Scale all dimensions to integer units (0.1 mm precision).
        2. Run FFD to obtain an upper bound on the sheet count and a
           warm-start placement hint.
        3. Build the CP-SAT model with optional interval variables for
           AddNoOverlap2D, plus assignment, sheet-usage, and symmetry-
           breaking constraints.
        4. Solve within the configured time limit.
        5. Extract and return the layout from solved variable values;
           fall back to the FFD result if CP-SAT fails.

        Parameters
        ----------
        labels   : Label instances pre-sorted by area.
        material : Material identifier for Sheet construction.

        Returns
        -------
        List of Sheet objects with ILP-optimal (or FFD-fallback) placements.
        """
        from ortools.sat.python import cp_model

        S = self.SCALE
        W = int(round(self.sheet_width  * S))
        H = int(round(self.sheet_height * S))

        # ── Prepare item data ──────────────────────────────────────────
        # items[i] = {'label': Label, 'oris': [(w_int, h_int, rot_deg), ...]}
        items = []
        for label in labels:
            w0 = int(round(label.width  * S))
            h0 = int(round(label.height * S))
            oris = []
            if w0 <= W and h0 <= H:
                oris.append((w0, h0, 0))
            if w0 != h0 and h0 <= W and w0 <= H:
                oris.append((h0, w0, 90))
            if not oris:
                print(f"  ⚠️  Label {label.width}×{label.height} mm "
                      f"doesn't fit — skipped")
                continue
            items.append({'label': label, 'oris': oris})

        if not items:
            return []

        n = len(items)

        # ── FFD upper bound & warm-start hint ─────────────────────────
        ffd = SimpleBinPacker(self.sheet_width, self.sheet_height)
        ffd_sheets = ffd._pack_material(labels, material)
        ub = len(ffd_sheets)
        print(f"  → FFD upper bound: {ub} sheet(s)")

        # ── Build CP-SAT model ────────────────────────────────────────
        model = cp_model.CpModel()

        # Variable containers
        a_var: Dict[Tuple, object] = {}   # (i, k, oi) -> BoolVar
        x_var: Dict[Tuple, object] = {}   # (i, k, oi) -> IntVar
        y_var: Dict[Tuple, object] = {}   # (i, k, oi) -> IntVar
        # Per-sheet interval lists for AddNoOverlap2D
        x_itvs = [[] for _ in range(ub)]
        y_itvs = [[] for _ in range(ub)]

        for i, item in enumerate(items):
            for k in range(ub):
                for oi, (w, h, _) in enumerate(item['oris']):
                    key = (i, k, oi)
                    a = model.NewBoolVar(f'a_{i}_{k}_{oi}')
                    x = model.NewIntVar(0, W - w, f'x_{i}_{k}_{oi}')
                    y = model.NewIntVar(0, H - h, f'y_{i}_{k}_{oi}')
                    xi = model.NewOptionalIntervalVar(x, w, x + w, a,
                                                     f'xi_{i}_{k}_{oi}')
                    yi = model.NewOptionalIntervalVar(y, h, y + h, a,
                                                     f'yi_{i}_{k}_{oi}')
                    a_var[key] = a
                    x_var[key] = x
                    y_var[key] = y
                    x_itvs[k].append(xi)
                    y_itvs[k].append(yi)

        # Constraint 1 — each item placed exactly once
        for i, item in enumerate(items):
            once = [a_var[(i, k, oi)]
                    for k in range(ub)
                    for oi in range(len(item['oris']))]
            model.AddExactlyOne(once)

        # Constraint 2 — no overlap per sheet
        for k in range(ub):
            model.AddNoOverlap2D(x_itvs[k], y_itvs[k])

        # Constraint 3 — sheet usage indicator
        sheet_used = [model.NewBoolVar(f'su_{k}') for k in range(ub)]
        for k in range(ub):
            for i, item in enumerate(items):
                for oi in range(len(item['oris'])):
                    model.Add(sheet_used[k] >= a_var[(i, k, oi)])

        # Constraint 4 — symmetry breaking (packed sheets are contiguous)
        for k in range(ub - 1):
            model.Add(sheet_used[k] >= sheet_used[k + 1])

        # Objective
        model.Minimize(sum(sheet_used))

        # ── Warm-start hints from FFD ──────────────────────────────────
        self._add_ffd_hints(model, ffd_sheets, items, a_var, x_var, y_var,
                            sheet_used, S)

        # ── Solve ─────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds  = self.time_limit
        solver.parameters.num_workers          = 8
        solver.parameters.log_search_progress  = False

        status = solver.Solve(model)
        status_name = solver.StatusName(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            n_used = int(round(solver.ObjectiveValue()))
            optimality = "optimal" if status == cp_model.OPTIMAL else "feasible"
            print(f"  → ILP solution: {n_used} sheet(s) "
                  f"[{optimality}] in {solver.WallTime():.1f}s")
            return self._extract_sheets(solver, items, a_var, x_var, y_var,
                                        ub, material, S)
        else:
            print(f"  ⚠️  CP-SAT returned {status_name} — using FFD solution")
            return ffd_sheets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_ffd_hints(self, model, ffd_sheets: List[Sheet], items: list,
                       a_var: dict, x_var: dict, y_var: dict,
                       sheet_used: list, S: int) -> None:
        """
        Seed the CP-SAT model with FFD placement positions as initial hints.

        Providing a known-feasible starting point lets the solver prove
        optimality faster by pruning branches that cannot beat the FFD cost.

        The hint strategy is:
          • For each item, look up its FFD sheet index and position.
          • Hint the matching (item, sheet, orientation) assignment variable
            to 1 and all others to 0.
          • Hint the x/y position variables with the FFD coordinates.
          • Hint each sheet_used variable to 1 if used by FFD, else 0.

        Parameters
        ----------
        model      : CpModel being built.
        ffd_sheets : Sheets produced by the FFD heuristic.
        items      : Item metadata list (same order as model variables).
        a_var      : Dict (i, k, oi) → BoolVar — assignment variables.
        x_var      : Dict (i, k, oi) → IntVar  — x-position variables.
        y_var      : Dict (i, k, oi) → IntVar  — y-position variables.
        sheet_used : List of BoolVar, one per candidate sheet.
        S          : Integer scale factor (mm → integer units).
        """
        # Build a queue of (label_id, sheet_idx, x_mm, y_mm, rot) from FFD
        ffd_queue: Dict[int, list] = defaultdict(list)
        for k, sheet in enumerate(ffd_sheets):
            for p in sheet.placements:
                ffd_queue[id(p.label)].append(
                    (k, int(round(p.x * S)), int(round(p.y * S)), p.rotation)
                )

        for i, item in enumerate(items):
            lid = id(item['label'])
            if lid not in ffd_queue or not ffd_queue[lid]:
                continue
            k_hint, x_hint, y_hint, rot_hint = ffd_queue[lid].pop(0)

            for oi, (w, h, rot) in enumerate(item['oris']):
                is_hint = (rot == rot_hint)
                key = (i, k_hint, oi)
                if key in a_var:
                    model.AddHint(a_var[key], 1 if is_hint else 0)
                    if is_hint:
                        model.AddHint(x_var[key], x_hint)
                        model.AddHint(y_var[key], y_hint)
                # All other sheets: hint inactive
                for k2 in range(len(ffd_sheets)):
                    if k2 == k_hint:
                        continue
                    key2 = (i, k2, oi)
                    if key2 in a_var:
                        model.AddHint(a_var[key2], 0)

        for k, su in enumerate(sheet_used):
            model.AddHint(su, 1 if k < len(ffd_sheets) else 0)

    def _extract_sheets(self, solver, items: list, a_var: dict, x_var: dict,
                        y_var: dict, ub: int, material: str, S: int) -> List[Sheet]:
        """
        Reconstruct Sheet/Placement objects from solved CP-SAT variable values.

        For every candidate sheet k, collects all items whose assignment
        variable a[i,k,oi] equals 1, reads their x/y positions, and builds
        the corresponding Placement objects.  Sheets with no placements
        (i.e. unused candidate slots) are omitted.

        Parameters
        ----------
        solver   : Solved CpSolver instance.
        items    : Item metadata list (same order as model variables).
        a_var    : Dict (i, k, oi) → BoolVar — assignment variables.
        x_var    : Dict (i, k, oi) → IntVar  — x-position variables.
        y_var    : Dict (i, k, oi) → IntVar  — y-position variables.
        ub       : Number of candidate sheets (upper bound from FFD).
        material : Material identifier for Sheet construction.
        S        : Integer scale factor used to convert back to mm.

        Returns
        -------
        List of non-empty Sheet objects in sheet-index order.
        """
        sheets: List[Sheet] = []
        for k in range(ub):
            placements: List[Placement] = []
            for i, item in enumerate(items):
                for oi, (w, h, rot) in enumerate(item['oris']):
                    key = (i, k, oi)
                    if key in a_var and solver.Value(a_var[key]):
                        x_mm = solver.Value(x_var[key]) / S
                        y_mm = solver.Value(y_var[key]) / S
                        placements.append(Placement(item['label'], x_mm, y_mm, rot))
                        break  # item placed — no need to check other orientations
            if placements:
                sheets.append(Sheet(self.sheet_width, self.sheet_height,
                                    placements, material))
        return sheets


# ============================================================================
# CSV PARSER
# ============================================================================

def parse_csv(filename: str) -> List[Label]:
    """
    Parse a CSV file into Label objects.

    The delimiter and quote character are detected automatically via
    ``csv.Sniffer``.  If sniffing fails the function falls back to
    semicolons (;) with double-quote (") quoting, which is the format
    produced by most European spreadsheet applications.

    The CSV must have a header row with at least the following columns
    (aliases accepted, case-insensitive):

    ============  ==========================================
    Column        Aliases
    ============  ==========================================
    QUANTITY      QTY, COUNT
    MATERIAL      MAT, MATERIAL TYPE
    TEXTDATA      TEXT, LABEL TEXT, CONTENT
    LABEL WIDTH   LABEL WIDTH(MM), WIDTH(MM), WIDTH
    LABEL HEIGHT  LABEL HEIGHT(MM), HEIGHT(MM), HEIGHT
    TEXT HEIGHT   TEXT HEIGHT(MM), TEXT SIZE(MM), FONT SIZE
    ============  ==========================================

    Multi-line text values are supported via quoted fields with embedded
    newlines.  Rows that cannot be parsed are skipped with a warning.

    Parameters
    ----------
    filename : Path to the input CSV file (UTF-8 encoded).

    Returns
    -------
    List of Label objects; empty if the file could not be read or the
    required columns are absent.
    """
    labels = []

    with open(filename, 'r', encoding='utf-8') as f:
        # ── Auto-detect dialect ───────────────────────────────────────
        sample = f.read(4096)
        f.seek(0)
        delimiter = ';'
        quotechar = '"'
        try:
            detected = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            delimiter = detected.delimiter
            quotechar = detected.quotechar or '"'
            print(f"  → Detected CSV dialect: delimiter={delimiter!r} "
                  f"quotechar={quotechar!r}")
        except csv.Error:
            reason = "file may be too short or ambiguous" if len(sample) < 512 else "content is ambiguous"
            print(f"  ⚠️  Warning: CSV dialect detection failed ({reason}).")
            print(f"  ↩  Falling back to default: delimiter={delimiter!r}  quotechar={quotechar!r}")
            print(f"       If parsing fails, check that your CSV uses one of: , ; \\t |")

        reader = csv.reader(f,
                            delimiter=delimiter,
                            quotechar=quotechar,
                            doublequote=True,
                            skipinitialspace=True)
        
        # Read header
        header = next(reader, None)
        if not header:
            print("Error: Empty CSV file")
            return []
        
        # Normalize header (strip quotes and whitespace, uppercase)
        header = [h.strip().strip('"').upper() for h in header]
        
        # Create field index mapping
        field_map = {}
        for i, field in enumerate(header):
            field_map[field] = i
        
        # Required fields with alternative names
        required_fields = {
            'QUANTITY': ['QUANTITY', 'QTY', 'COUNT'],
            'MATERIAL': ['MATERIAL', 'MAT', 'MATERIAL TYPE'],
            'TEXTDATA': ['TEXTDATA', 'TEXT', 'LABEL TEXT', 'CONTENT'],
            'WIDTH': ['LABEL WIDTH(MM)', 'WIDTH(MM)', 'WIDTH', 'LABEL WIDTH'],
            'HEIGHT': ['LABEL HEIGHT(MM)', 'HEIGHT(MM)', 'HEIGHT', 'LABEL HEIGHT'],
            'TEXT_HEIGHT': ['TEXT HEIGHT(MM)', 'TEXT SIZE(MM)', 'TEXT HEIGHT', 'FONT SIZE(MM)', 'FONT SIZE']
        }
        
        # Find field indices
        indices = {}
        for key, possible_names in required_fields.items():
            for name in possible_names:
                if name in field_map:
                    indices[key] = field_map[name]
                    break
            if key not in indices:
                print(f"Error: Required field not found. Looking for one of: {possible_names}")
                print(f"Available fields: {header}")
                return []
        
        # Parse rows
        row_num = 1
        for row in reader:
            row_num += 1
            
            # Skip empty rows
            if not row or all(not cell.strip() for cell in row):
                continue
            
            try:
                # Extract fields
                quantity = int(row[indices['QUANTITY']].strip())
                material = row[indices['MATERIAL']].strip()
                textdata = row[indices['TEXTDATA']].strip()
                # Normalise newlines: DOS \r\n → \n, then expand literal \n escape
                textdata = textdata.replace('\r\n', '\n').replace('\r', '\n')
                textdata = textdata.replace('\\n', '\n')
                width = float(row[indices['WIDTH']].strip())
                height = float(row[indices['HEIGHT']].strip())
                text_height = float(row[indices['TEXT_HEIGHT']].strip())
                
                # Create label
                label = Label(
                    material=material,
                    width=width,
                    height=height,
                    text=textdata,
                    text_height=text_height,
                    quantity=quantity
                )
                labels.append(label)
                
                # Preview lines of text on one line (for logging)
                linetext = textdata.replace('\n','\\n')[:40]
                if len(textdata.replace('\n','\\n')) > 40:
                    linetext += "..."
                print(f"  Row {row_num}: {quantity}x {material} ({width}x{height}mm) - '{linetext}'")
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid row {row_num}: {e}")
                print(f"  Row data: {row}")
                continue
    
    return labels

# ============================================================================
# SVGGenerator
# ============================================================================
class SVGGenerator:
    """
    Render a packed Sheet as an SVG file suitable for laser processing.

    The generated SVG contains three named groups / layers in draw order
    (back to front):

    =========  =======  ================================================
    Group id   Colour   Purpose
    =========  =======  ================================================
    cut_lines  red      0.1 mm stroke rectangle at each label boundary.
                        Laser cuts along these lines.  (back layer)
    borders    amber    0.8 mm solid fill frame around each label.
                        Laser engraves the border decoration.
    text       blue     Label text converted to filled path outlines.
                        Laser engraves the text shapes.  (front layer)
    =========  =======  ================================================

    Within each group elements are ordered by coordinate position
    (top-to-bottom, then left-to-right).

    Adjacent/touching/overlapping rectangles in the borders and cut_lines
    groups are merged before output to reduce the total path count and
    improve laser throughput.

    Text is rendered via fontTools (TrueType outlines) if a suitable font
    can be located, otherwise falls back to placeholder rectangles.
    """
    
    # Colors
    COLOR_CUT = "#f44336"      # Red for cutting
    COLOR_ENGRAVE = "#ffc107"   # Amber for engraving (border)
    COLOR_TEXT = "#0000ff"      # Blue for text
    BORDER_WIDTH = 0.8          # mm - border etching width
    
    def __init__(self, sheet: Sheet) -> None:
        """
        Prepare the generator for a specific sheet.

        Parameters
        ----------
        sheet : The Sheet (with all Placements) to render.
        """
        self.sheet = sheet
        self._init_font()
        
    def _init_font(self) -> None:
        """
        Locate and load a TrueType font for text-to-path conversion.

        Searches a platform-specific list of candidate paths for the first
        available font file and loads it with fontTools.  Sets self.font to
        the loaded TTFont object, or to None if no font could be found or
        fontTools is not installed.

        Searched paths (in order)
        -------------------------
        * arial.ttf                        (current working directory)
        * Liberation Sans  (Linux system)
        * DejaVu Sans      (Linux system)
        * Helvetica        (macOS system)
        * Arial            (Windows system)
        """
        try:
            from fontTools.ttLib import TTFont
            import os
            
            # Try to load Arial font
            font_paths = [
                'arial.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/System/Library/Fonts/Helvetica.ttc',
                'C:\\Windows\\Fonts\\arial.ttf',
            ]
            
            self.font = None
            self.font_path = None
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        self.font = TTFont(font_path)
                        self.font_path = font_path
                        print(f"  → Loaded font: {font_path}")
                        break
                    except Exception as e:
                        continue
            
            if not self.font:
                print(f"  ⚠️  Warning: No suitable font found, text will use fallback rendering")
                
        except ImportError:
            print(f"  ⚠️  Warning: fontTools not installed. Install with: pip install fonttools")
            self.font = None
    
    def generate(self) -> str:
        """
        Render the sheet to an SVG string.

        Processing steps
        ----------------
        1. Collect four border rectangles and four cut-line rectangles for
           each label placement.
        2. Convert each label’s text to a single merged SVG path.
        3. Merge collinear/adjacent/overlapping border and cut-line
           rectangles to reduce element count.
        4. Assemble the final SVG with three named groups and inject a
           metadata comment block after the XML declaration.

        Returns
        -------
        Complete SVG document as a UTF-8 string.  Falls back to a minimal
        error SVG if svgwrite is not installed.
        """
        
        try:
            import svgwrite
        except ImportError:
            print(f"  ⚠️  Error: svgwrite not installed. Install with: pip install svgwrite")
            return self._generate_fallback()
        
        # Create SVG document
        dwg = svgwrite.Drawing(size=(f"{self.sheet.width}mm", f"{self.sheet.height}mm"),
                              viewBox=f"0 0 {self.sheet.width} {self.sheet.height}")
        
        # Add styles
        style_text = f"""
            .cut {{ fill: none; stroke: {self.COLOR_CUT}; stroke-width: 0.1; }}
            .engrave {{ fill: {self.COLOR_ENGRAVE}; stroke: none; }}
            .text-path {{ fill: {self.COLOR_TEXT}; stroke: none; }}
        """
        dwg.defs.add(dwg.style(style_text))
        
        # Collect all borders and cut lines
        all_borders = []
        all_cut_lines = []
        all_text_paths = []
        
        for i, placement in enumerate(self.sheet.placements):
            borders, cut_lines, text_paths = self._extract_placement_elements(placement, i)
            all_borders.extend(borders)
            all_cut_lines.extend(cut_lines)
            for tp in text_paths:
                tp['_sort_y'] = placement.y
                tp['_sort_x'] = placement.x
            all_text_paths.extend(text_paths)
        
        # Optimize borders and cut lines
        print(f"  → Optimizing geometry...")
        print(f"     Before: {len(all_borders)} border rectangles, {len(all_cut_lines)} cut lines")
        
        optimized_borders = self._optimize_rectangles(all_borders)
        optimized_cut_lines = self._optimize_cut_lines(all_cut_lines)
        
        print(f"     After:  {len(optimized_borders)} border rectangles, {len(optimized_cut_lines)} cut lines")
        print(f"     Text:   {len(all_text_paths)} text paths generated (one path per label)")

        # Sort each group by coordinates (top-to-bottom, left-to-right)
        optimized_borders.sort(key=lambda r: (r['y'], r['x']))
        optimized_cut_lines.sort(key=lambda l: (min(l['y1'], l['y2']), min(l['x1'], l['x2'])))
        all_text_paths.sort(key=lambda tp: (tp.get('_sort_y', 0), tp.get('_sort_x', 0)))

        # Add cut lines group (back layer)
        cut_lines_group = dwg.g(id='cut_lines')
        for cut_line in optimized_cut_lines:
            cut_lines_group.add(dwg.line(
                start=(cut_line['x1'], cut_line['y1']),
                end=(cut_line['x2'], cut_line['y2']),
                class_='cut'
            ))
        dwg.add(cut_lines_group)

        # Add borders group (middle layer)
        borders_group = dwg.g(id='borders')
        for border in optimized_borders:
            borders_group.add(dwg.rect(
                insert=(border['x'], border['y']),
                size=(border['width'], border['height']),
                class_='engrave'
            ))
        dwg.add(borders_group)

        # Add text paths group (front layer)
        text_group = dwg.g(id='text')
        for text_path_data in all_text_paths:
            if text_path_data and text_path_data.get('d'):
                path_element = dwg.path(
                    d=text_path_data['d'],
                    class_='text-path'
                )
                # No transform attribute - coordinates already transformed
                if text_path_data.get('transform') and text_path_data['transform']:
                    path_element['transform'] = text_path_data['transform']
                text_group.add(path_element)
        dwg.add(text_group)
        
        # Get SVG string
        svg_string = dwg.tostring()
        
        # Insert metadata comments after XML declaration
        metadata_comment = f"""
<!-- Laser Cut Sheet Generator -->
<!-- Material: {self.sheet.material} -->
<!-- Pieces: {len(self.sheet.placements)} -->
<!-- Efficiency: {self.sheet.efficiency:.1f}% -->
<!-- Text: Merged paths (one path per text line) -->
"""
        
        # Find position after XML declaration
        if svg_string.startswith('<?xml'):
            xml_decl_end = svg_string.find('?>') + 2
            svg_string = svg_string[:xml_decl_end] + metadata_comment + svg_string[xml_decl_end:]
        else:
            # No XML declaration, add at beginning
            svg_string = metadata_comment + svg_string
        
        return svg_string
    
    def _generate_fallback(self) -> str:
        """
        Produce a minimal error SVG when svgwrite is unavailable.

        Returns a bare SVG with the correct viewport dimensions and a
        human-readable error message rendered as an SVG <text> element.

        Returns
        -------
        Minimal SVG string with an inline error notice.
        """
        
        svg_parts = []
        
        svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg viewBox="0 0 {self.sheet.width} {self.sheet.height}"
     width="{self.sheet.width}mm"
     height="{self.sheet.height}mm"
     xmlns="http://www.w3.org/2000/svg">
<!-- Laser Cut Sheet Generator -->
<!-- Material: {self.sheet.material} -->
<!-- Pieces: {len(self.sheet.placements)} -->
<!-- Efficiency: {self.sheet.efficiency:.1f}% -->
<defs>
  <style>
    .cut {{ fill: none; stroke: {self.COLOR_CUT}; stroke-width: 0.1; }}
    .engrave {{ fill: {self.COLOR_ENGRAVE}; stroke: none; }}
    .text-path {{ fill: {self.COLOR_TEXT}; stroke: none; }}
  </style>
</defs>
<g id="error">
  <text x="50" y="50" font-size="12" fill="red">Error: svgwrite not installed</text>
</g>
</svg>''')
        
        return '\n'.join(svg_parts)
    
    def _extract_placement_elements(self, p: Placement, index: int) -> tuple:
        """
        Decompose one Placement into its three element lists.

        Generates the raw (un-merged) geometry for a single label:

        * **borders**   — four axis-aligned filled rectangles forming the
          0.8 mm engraved frame (top, bottom, left, right strips).
        * **cut_lines** — four 0.1 mm thin rectangles representing the
          cutting outline (same topology as borders but much thinner).
        * **text_paths** — a single merged SVG path for all text on the
          label, or an empty list when the label has no text.

        Rotation is handled by swapping width/height; the (x, y) origin
        is always the placement’s top-left corner on the sheet.

        Parameters
        ----------
        p     : Placement whose geometry is to be extracted.
        index : Zero-based index of this placement on the sheet (unused
                internally but available for debugging).

        Returns
        -------
        Tuple of (borders, cut_lines, text_paths) — each a list of dicts.
        """
        
        # Get dimensions (considering rotation)
        if p.rotation == 0:
            width = p.label.width
            height = p.label.height
            x, y = p.x, p.y
        else:
            width = p.label.height
            height = p.label.width
            x, y = p.x, p.y
        
        borders = []
        cut_lines = []
        text_paths = []
        
        # Cut lines as true line segments (x1,y1 → x2,y2)
        cut_lines.extend([
            {'x1': x,         'y1': y,          'x2': x + width,  'y2': y,          'orientation': 'horizontal'},  # top
            {'x1': x,         'y1': y + height, 'x2': x + width,  'y2': y + height, 'orientation': 'horizontal'},  # bottom
            {'x1': x,         'y1': y,          'x2': x,          'y2': y + height, 'orientation': 'vertical'},    # left
            {'x1': x + width, 'y1': y,          'x2': x + width,  'y2': y + height, 'orientation': 'vertical'},    # right
        ])
        
        # Border rectangles
        borders.extend([
            {'x': x, 'y': y, 'width': width, 'height': self.BORDER_WIDTH, 'orientation': 'horizontal'},
            {'x': x, 'y': y + height - self.BORDER_WIDTH, 'width': width, 'height': self.BORDER_WIDTH, 'orientation': 'horizontal'},
            {'x': x, 'y': y, 'width': self.BORDER_WIDTH, 'height': height, 'orientation': 'vertical'},
            {'x': x + width - self.BORDER_WIDTH, 'y': y, 'width': self.BORDER_WIDTH, 'height': height, 'orientation': 'vertical'}
        ])
        
        # Convert text to paths.
        # For rotated labels the text is generated in the label's *original*
        # (unrotated) coordinate frame, centred on the same point as the placed
        # bounding box, and then an SVG rotate(90) transform is attached so the
        # glyphs stay upright relative to the label face.
        if p.label.text:
            if p.rotation == 0:
                text_paths.extend(
                    self._text_to_paths(p.label.text, x, y, width, height,
                                        p.label.text_height)
                )
            else:
                # Centre of the placed (rotated) bounding box
                cx = p.x + width  / 2   # width  = label.height after swap
                cy = p.y + height / 2   # height = label.width  after swap
                # Synthetic top-left of the *original* label rectangle,
                # centred on (cx, cy) with original dimensions restored
                x_orig = cx - p.label.width  / 2
                y_orig = cy - p.label.height / 2
                raw = self._text_to_paths(p.label.text,
                                          x_orig, y_orig,
                                          p.label.width, p.label.height,
                                          p.label.text_height)
                for tp in raw:
                    tp['transform'] = f'rotate(90,{cx},{cy})'
                text_paths.extend(raw)

        return borders, cut_lines, text_paths
    
    def _text_to_paths(self, text: str, x: float, y: float,
                       width: float, height: float,
                       text_height: float) -> list:
        """
        Convert all text lines for one label into a single merged SVG path.

        Layout algorithm
        ----------------
        * Split *text* on newlines; blank lines are discarded.
        * Compute the total block height and centre it vertically inside
          the label bounding box.
        * For each line, centre it horizontally in the label width and
          call _create_text_path to obtain path data.
        * Concatenate all per-line path data strings into one compound
          path string (space-separated).

        Parameters
        ----------
        text        : Raw label text (may contain \\n line separators).
        x           : Label left edge on the sheet (mm).
        y           : Label top edge on the sheet (mm).
        width       : Label width in the placed orientation (mm).
        height      : Label height in the placed orientation (mm).
        text_height : Target cap-height for each text line (mm).

        Returns
        -------
        List containing at most one path dict with key ‘d’ holding the
        merged SVG path commands, or an empty list if no text is produced.
        """
        
        text_lines = [line for line in text.split('\n') if line.strip()]
        
        if not text_lines:
            return []
        
        num_lines = len(text_lines)
        line_height = text_height * 1.2
        total_text_height = (num_lines - 1) * line_height + text_height
        start_y = y + (height - total_text_height) / 2
        
        all_d_parts = []
        
        for i, line in enumerate(text_lines):
            text_x = x + width / 2
            text_y = start_y + i * line_height + text_height
            
            line_paths = self._create_text_path(line.strip(), text_x, text_y, text_height)
            for p in line_paths:
                if p and p.get('d'):
                    all_d_parts.append(p['d'])
        
        if all_d_parts:
            return [{'d': ' '.join(all_d_parts), 'transform': ''}]
        
        return []
    
    def _create_text_path(self, text: str, x: float, y: float, size: float) -> list:
        """
        Render one line of text to a single merged SVG path via fontTools.

        Algorithm
        ---------
        1. Walk *text* character by character, resolving each code point
           through the font’s best cmap to a glyph name.
        2. Accumulate total advance width to compute the centred start X.
        3. For each glyph, construct an affine matrix::

               (scale, 0, 0, −scale, current_x, y)

           where *scale* = size / units_per_em.  The negative y-scale
           converts from the font’s y-up coordinate system to SVG’s y-down
           system.  Feed the glyph through a TransformPen wrapping a
           shared SVGPathPen so that all characters land in one path string.
        4. Return the serialised path commands from SVGPathPen.getCommands().

        Falls back to _create_fallback_text_path if no font is loaded or
        an exception occurs during rendering.

        Parameters
        ----------
        text : Single line of text to render (no newlines).
        x    : Horizontal centre of the text block on the sheet (mm).
        y    : Baseline position on the sheet (mm).
        size : Target cap-height / em-size in mm.

        Returns
        -------
        List with one path dict ({'d': ‘…’, 'transform': ‘’}) on success,
        or an empty list if no visible glyphs were drawn.
        """
        
        if not self.font:
            return self._create_fallback_text_path(text, x, y, size)
        
        try:
            from fontTools.pens.svgPathPen import SVGPathPen
            from fontTools.pens.transformPen import TransformPen
            
            # Get font metrics
            units_per_em = self.font['head'].unitsPerEm
            scale = size / units_per_em
            
            # Get character map
            cmap = self.font.getBestCmap()
            if not cmap:
                return self._create_fallback_text_path(text, x, y, size)
            
            glyph_set = self.font.getGlyphSet()
            
            # First pass: calculate total advance width for centering
            total_width = 0.0
            char_glyphs = []
            for char in text:
                glyph_name = cmap.get(ord(char))
                if glyph_name and glyph_name in glyph_set:
                    glyph = glyph_set[glyph_name]
                    advance = glyph.width * scale
                else:
                    glyph = None
                    advance = size * 0.5
                char_glyphs.append((glyph, advance))
                total_width += advance
            
            # Second pass: draw each glyph with TransformPen into a single SVGPathPen
            # Use one shared SVGPathPen so all glyphs end up in a single path string
            merged_pen = SVGPathPen(glyph_set)
            current_x = x - total_width / 2
            
            for glyph, advance in char_glyphs:
                if glyph is not None:
                    # Affine matrix: scale x, flip y, translate to (current_x, y)
                    # Maps font point (fx, fy) → (current_x + fx*scale, y - fy*scale)
                    transform_matrix = (scale, 0, 0, -scale, current_x, y)
                    t_pen = TransformPen(merged_pen, transform_matrix)
                    glyph.draw(t_pen)
                current_x += advance
            
            path_commands = merged_pen.getCommands()
            
            if path_commands:
                return [{'d': path_commands, 'transform': ''}]
            
            return []
            
        except Exception as e:
            print(f"  ⚠️  Warning: Text-to-path conversion failed: {e}")
            return self._create_fallback_text_path(text, x, y, size )
    
    def _create_fallback_text_path(self, text: str, x: float, y: float,
                                   size: float) -> list:
        """
        Produce placeholder text geometry when no font is available.

        Draws a solid filled rectangle for every non-space character,
        positioned as if printed in a monospaced font with:
          * character width  = size × 0.6
          * character spacing = size × 0.1

        All character rectangles are merged into a single compound SVG
        path string (space-separated M…Z sub-paths).

        Parameters
        ----------
        text : Single line of text (no newlines).
        x    : Horizontal centre of the text block on the sheet (mm).
        y    : Baseline position on the sheet (mm).
        size : Target character height in mm.

        Returns
        -------
        List with one path dict, or an empty list if *text* is blank.
        """
        
        if not text:
            return []
        
        char_width = size * 0.6
        spacing = size * 0.1
        total_width = len(text) * (char_width + spacing) - spacing
        start_x = x - total_width / 2
        
        # Collect all character rectangles into one path
        all_rects = []
        
        for i, char in enumerate(text):
            if char == ' ':
                continue
            
            char_x = start_x + i * (char_width + spacing)
            char_y = y - size * 0.8
            
            # Rectangle path for this character
            rect_path = (
                f"M {char_x:.3f},{char_y:.3f} "
                f"L {char_x + char_width:.3f},{char_y:.3f} "
                f"L {char_x + char_width:.3f},{char_y + size:.3f} "
                f"L {char_x:.3f},{char_y + size:.3f} "
                f"Z"
            )
            all_rects.append(rect_path)
        
        # Merge all rectangles into one path
        if all_rects:
            merged_path = ' '.join(all_rects)
            return [{
                'd': merged_path,
                'transform': ''
            }]
        
        return []
    
    def _optimize_rectangles(self, rectangles: list) -> list:
        """
        Iteratively merge a list of axis-aligned rectangles until stable.

        Three merge strategies are attempted for every unique pair on each
        iteration pass:

        * **end-to-end**   — two same-orientation strips that share an edge
          along their major axis are concatenated into one longer strip.
        * **side-by-side** — two same-orientation strips that are flush
          along their minor axis are combined, potentially changing the
          dominant orientation.
        * **overlapping**  — two thin strips at the same position that
          overlap or touch are replaced by their bounding rectangle.

        Iteration halts when a full pass produces no merges, or after 100
        passes (safety guard against infinite loops).

        Parameters
        ----------
        rectangles : List of rect dicts with keys x, y, width, height,
                     orientation (‘horizontal’ | ‘vertical’).

        Returns
        -------
        Reduced list of merged rect dicts.
        """
        
        if not rectangles:
            return []
        
        TOLERANCE = 0.01  # mm tolerance for alignment
        
        # Convert to list of dicts for easier manipulation
        rects = [dict(r) for r in rectangles]
        
        # Iteratively merge until no more merges possible
        merged_any = True
        iteration = 0
        
        while merged_any and iteration < 100:  # Safety limit
            merged_any = False
            iteration += 1
            new_rects = []
            processed = set()
            
            for i, rect1 in enumerate(rects):
                if i in processed:
                    continue
                
                # Try to merge rect1 with any other rectangle
                merged = dict(rect1)
                processed.add(i)
                
                for j, rect2 in enumerate(rects):
                    if j in processed or i == j:
                        continue
                    
                    # Try different merge strategies
                    new_merged = (
                        self._try_merge_end_to_end(merged, rect2, TOLERANCE) or
                        self._try_merge_side_by_side(merged, rect2, TOLERANCE) or
                        self._try_merge_overlapping(merged, rect2, TOLERANCE)
                    )
                    
                    if new_merged:
                        merged = new_merged
                        processed.add(j)
                        merged_any = True
                
                new_rects.append(merged)
            
            rects = new_rects
        
        return rects
    
    def _try_merge_end_to_end(self, rect1: dict, rect2: dict,
                              tolerance: float) -> Optional[dict]:
        """
        Merge two rectangles that are collinear and touching end-to-end.

        Horizontal pair: same Y and height, right edge of one touching
        the left edge of the other → merged into one wider horizontal rect.

        Vertical pair: same X and width, bottom edge of one touching the
        top of the other → merged into one taller vertical rect.

        Parameters
        ----------
        rect1     : First rectangle dict.
        rect2     : Second rectangle dict.
        tolerance : Positional tolerance in mm for ‘touching’ comparisons.

        Returns
        -------
        Merged rect dict if the merge is applicable, else None.
        """
        
        orientation1 = rect1.get('orientation')
        orientation2 = rect2.get('orientation')
        
        if orientation1 != orientation2:
            return None
        
        if orientation1 == 'horizontal':
            # Same Y and height, adjacent on X axis
            if (abs(rect1['y'] - rect2['y']) < tolerance and
                abs(rect1['height'] - rect2['height']) < tolerance):
                
                # rect2 is to the right of rect1
                if abs(rect1['x'] + rect1['width'] - rect2['x']) < tolerance:
                    return {
                        'x': rect1['x'],
                        'y': rect1['y'],
                        'width': rect1['width'] + rect2['width'],
                        'height': rect1['height'],
                        'orientation': 'horizontal'
                    }
                # rect1 is to the right of rect2
                elif abs(rect2['x'] + rect2['width'] - rect1['x']) < tolerance:
                    return {
                        'x': rect2['x'],
                        'y': rect1['y'],
                        'width': rect1['width'] + rect2['width'],
                        'height': rect1['height'],
                        'orientation': 'horizontal'
                    }
        
        elif orientation1 == 'vertical':
            # Same X and width, adjacent on Y axis
            if (abs(rect1['x'] - rect2['x']) < tolerance and
                abs(rect1['width'] - rect2['width']) < tolerance):
                
                # rect2 is below rect1
                if abs(rect1['y'] + rect1['height'] - rect2['y']) < tolerance:
                    return {
                        'x': rect1['x'],
                        'y': rect1['y'],
                        'width': rect1['width'],
                        'height': rect1['height'] + rect2['height'],
                        'orientation': 'vertical'
                    }
                # rect1 is below rect2
                elif abs(rect2['y'] + rect2['height'] - rect1['y']) < tolerance:
                    return {
                        'x': rect1['x'],
                        'y': rect2['y'],
                        'width': rect1['width'],
                        'height': rect1['height'] + rect2['height'],
                        'orientation': 'vertical'
                    }
        
        return None
    
    def _try_merge_side_by_side(self, rect1: dict, rect2: dict,
                                tolerance: float) -> Optional[dict]:
        """
        Merge two strips that are flush along their minor axis.

        Two horizontal strips with identical X and width that touch
        vertically are merged into a block (orientation becomes vertical).

        Two vertical strips with identical Y and height that touch
        horizontally are merged into a block (orientation becomes horizontal).

        Parameters
        ----------
        rect1     : First rectangle dict.
        rect2     : Second rectangle dict.
        tolerance : Positional tolerance in mm.

        Returns
        -------
        Merged rect dict if applicable, else None.
        """
        
        # Check if rectangles are axis-aligned and have matching dimensions
        
        # Case 1: Both horizontal, same X and width, touching on Y
        if (rect1.get('orientation') == 'horizontal' and 
            rect2.get('orientation') == 'horizontal'):
            
            if (abs(rect1['x'] - rect2['x']) < tolerance and
                abs(rect1['width'] - rect2['width']) < tolerance):
                
                # Check if touching on Y axis
                if abs(rect1['y'] + rect1['height'] - rect2['y']) < tolerance:
                    # rect2 is below rect1
                    return {
                        'x': rect1['x'],
                        'y': rect1['y'],
                        'width': rect1['width'],
                        'height': rect1['height'] + rect2['height'],
                        'orientation': 'vertical'  # Changed orientation!
                    }
                elif abs(rect2['y'] + rect2['height'] - rect1['y']) < tolerance:
                    # rect1 is below rect2
                    return {
                        'x': rect1['x'],
                        'y': rect2['y'],
                        'width': rect1['width'],
                        'height': rect1['height'] + rect2['height'],
                        'orientation': 'vertical'
                    }
        
        # Case 2: Both vertical, same Y and height, touching on X
        if (rect1.get('orientation') == 'vertical' and 
            rect2.get('orientation') == 'vertical'):
            
            if (abs(rect1['y'] - rect2['y']) < tolerance and
                abs(rect1['height'] - rect2['height']) < tolerance):
                
                # Check if touching on X axis
                if abs(rect1['x'] + rect1['width'] - rect2['x']) < tolerance:
                    # rect2 is to the right of rect1
                    return {
                        'x': rect1['x'],
                        'y': rect1['y'],
                        'width': rect1['width'] + rect2['width'],
                        'height': rect1['height'],
                        'orientation': 'horizontal'  # Changed orientation!
                    }
                elif abs(rect2['x'] + rect2['width'] - rect1['x']) < tolerance:
                    # rect1 is to the right of rect2
                    return {
                        'x': rect2['x'],
                        'y': rect1['y'],
                        'width': rect1['width'] + rect2['width'],
                        'height': rect1['height'],
                        'orientation': 'horizontal'
                    }
        
        return None
    
    def _try_merge_overlapping(self, rect1: dict, rect2: dict,
                               tolerance: float) -> Optional[dict]:
        """
        Replace two overlapping or touching thin strips with their bounding rect.

        Only merges strips that are “thin” in one dimension (height or width
        < 2 × BORDER_WIDTH) and share the same Y (horizontal) or X (vertical)
        coordinate within *tolerance*.  This prevents unintentional merging
        of non-aligned border pieces from different label rows or columns.

        Parameters
        ----------
        rect1     : First rectangle dict.
        rect2     : Second rectangle dict.
        tolerance : Positional and size tolerance in mm.

        Returns
        -------
        Bounding-box rect dict if both conditions are met, else None.
        """
        
        # Check if rectangles overlap or touch
        x1_min, y1_min = rect1['x'], rect1['y']
        x1_max = rect1['x'] + rect1['width']
        y1_max = rect1['y'] + rect1['height']
        
        x2_min, y2_min = rect2['x'], rect2['y']
        x2_max = rect2['x'] + rect2['width']
        y2_max = rect2['y'] + rect2['height']
        
        # Check if they overlap or touch (with tolerance)
        x_overlap = not (x1_max < x2_min - tolerance or x2_max < x1_min - tolerance)
        y_overlap = not (y1_max < y2_min - tolerance or y2_max < y1_min - tolerance)
        
        if not (x_overlap and y_overlap):
            return None
        
        # Check if one dimension is very similar (thin rectangles aligned)
        width_similar = abs(rect1['width'] - rect2['width']) < tolerance
        height_similar = abs(rect1['height'] - rect2['height']) < tolerance
        
        # Only merge if rectangles are thin and aligned
        # (both are border-width rectangles in same orientation)
        
        # Horizontal borders (thin height, similar width range)
        if (rect1['height'] < 2 * self.BORDER_WIDTH and 
            rect2['height'] < 2 * self.BORDER_WIDTH):
            
            # Check if they're at the same Y level (touching/overlapping horizontally)
            if abs(y1_min - y2_min) < tolerance and height_similar:
                # Merge horizontally
                new_x = min(x1_min, x2_min)
                new_width = max(x1_max, x2_max) - new_x
                
                return {
                    'x': new_x,
                    'y': y1_min,
                    'width': new_width,
                    'height': rect1['height'],
                    'orientation': 'horizontal'
                }
        
        # Vertical borders (thin width, similar height range)
        if (rect1['width'] < 2 * self.BORDER_WIDTH and 
            rect2['width'] < 2 * self.BORDER_WIDTH):
            
            # Check if they're at the same X level (touching/overlapping vertically)
            if abs(x1_min - x2_min) < tolerance and width_similar:
                # Merge vertically
                new_y = min(y1_min, y2_min)
                new_height = max(y1_max, y2_max) - new_y
                
                return {
                    'x': x1_min,
                    'y': new_y,
                    'width': rect1['width'],
                    'height': new_height,
                    'orientation': 'vertical'
                }
        
        return None
    
    def _optimize_cut_lines(self, cut_lines: list) -> list:
        """
        Merge collinear, adjacent, or overlapping cut-line segments.

        Horizontal segments sharing the same Y are merged when their X
        ranges touch or overlap.  Vertical segments sharing the same X are
        merged when their Y ranges touch or overlap.  The process repeats
        until no further merges are possible.

        Parameters
        ----------
        cut_lines : List of line dicts with keys x1, y1, x2, y2,
                    orientation ('horizontal' | 'vertical').

        Returns
        -------
        Reduced list of merged line dicts.
        """
        if not cut_lines:
            return []

        TOLERANCE = 0.01  # mm

        lines = [dict(l) for l in cut_lines]
        merged_any = True
        iteration = 0

        while merged_any and iteration < 100:
            merged_any = False
            iteration += 1
            new_lines = []
            used = set()

            for i, a in enumerate(lines):
                if i in used:
                    continue
                merged = False
                for j, b in enumerate(lines):
                    if j <= i or j in used:
                        continue
                    if a['orientation'] != b['orientation']:
                        continue
                    if a['orientation'] == 'horizontal':
                        # Same Y?
                        if abs(a['y1'] - b['y1']) > TOLERANCE:
                            continue
                        # Ranges touch or overlap?
                        a_min, a_max = min(a['x1'], a['x2']), max(a['x1'], a['x2'])
                        b_min, b_max = min(b['x1'], b['x2']), max(b['x1'], b['x2'])
                        if b_min <= a_max + TOLERANCE and b_max >= a_min - TOLERANCE:
                            new_lines.append({
                                'x1': min(a_min, b_min), 'y1': a['y1'],
                                'x2': max(a_max, b_max), 'y2': a['y1'],
                                'orientation': 'horizontal',
                            })
                            used.add(i); used.add(j)
                            merged_any = merged = True
                            break
                    else:  # vertical
                        # Same X?
                        if abs(a['x1'] - b['x1']) > TOLERANCE:
                            continue
                        # Ranges touch or overlap?
                        a_min, a_max = min(a['y1'], a['y2']), max(a['y1'], a['y2'])
                        b_min, b_max = min(b['y1'], b['y2']), max(b['y1'], b['y2'])
                        if b_min <= a_max + TOLERANCE and b_max >= a_min - TOLERANCE:
                            new_lines.append({
                                'x1': a['x1'], 'y1': min(a_min, b_min),
                                'x2': a['x1'], 'y2': max(a_max, b_max),
                                'orientation': 'vertical',
                            })
                            used.add(i); used.add(j)
                            merged_any = merged = True
                            break
                if not merged and i not in used:
                    new_lines.append(a)
                    used.add(i)

            lines = new_lines

        return lines


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_laser_sheets(csv_file: str, sheet_width: float, sheet_height: float,
                         output_prefix: str = "output/cutsheet",
                         solver: str = "ilp",
                         time_limit: int = 60) -> List[str]:
    """
    Full pipeline: parse CSV → bin-pack → emit SVG sheets.

    Reads label definitions from *csv_file*, groups them by material, solves
    the 2-D cutting-stock problem for each material group, and renders one
    SVG file per output sheet.

    Parameters
    ----------
    csv_file : str
        Path to the semicolon-delimited CSV with label definitions.
    sheet_width : float
        Stock sheet width in mm.
    sheet_height : float
        Stock sheet height in mm.
    output_prefix : str, optional
        Filename prefix for output SVG files (default ``"cutsheet"``).
    solver : {'ilp', 'ffd'}
        ``'ilp'`` — OR-Tools CP-SAT optimal 2-D bin packing (requires
        the *ortools* package; falls back to FFD if unavailable).
        ``'ffd'`` — First-Fit Decreasing heuristic (fast, no extra deps).
    time_limit : int, optional
        CP-SAT wall-clock time limit in seconds per material group
        (ILP solver only, default 60).

    Returns
    -------
    List[str]
        Sorted list of generated SVG filenames.
    """

    print("=" * 70)
    print("LASER CUT SHEET GENERATOR")
    print("=" * 70)
    print(f"\nReading labels from: {csv_file}")
    print(f"Stock sheet size: {sheet_width} x {sheet_height} mm")
    print()

    labels = parse_csv(csv_file)

    if not labels:
        print("\n❌ Error: No valid labels found in CSV")
        return []

    print(f"\n✅ Found {len(labels)} label type(s)")
    total_pieces = sum(l.quantity for l in labels)
    total_area   = sum(l.area * l.quantity for l in labels)
    print(f"✅ Total pieces to cut: {total_pieces}")
    print(f"✅ Total area needed: {total_area:.1f} mm²")

    # Solve CSP
    print(f"\n{'─' * 70}")
    print("SOLVING CUTTING STOCK PROBLEM (CSP)")
    print(f"{'─' * 70}")

    if solver == "ilp":
        print("Algorithm: ILP — OR-Tools CP-SAT (exact 2-D bin packing)")
        print(f"Time limit: {time_limit}s per material group")
        packer = ILPBinPacker(sheet_width, sheet_height, time_limit=time_limit)
    else:
        print("Algorithm: FFD — First-Fit Decreasing heuristic (skyline strategy)")
        packer = SimpleBinPacker(sheet_width, sheet_height)
    print()

    sheets = packer.pack(labels)
    
    sheet_area = sheet_width * sheet_height
    total_sheet_area = len(sheets) * sheet_area
    overall_efficiency = (total_area / total_sheet_area * 100) if total_sheet_area > 0 else 0
    
    print(f"✅ Solution found: {len(sheets)} sheet(s) required")
    print(f"✅ Overall material efficiency: {overall_efficiency:.1f}%")
    
    # Generate SVG files
    print(f"\n{'─' * 70}")
    print("GENERATING SVG FILES")
    print(f"{'─' * 70}")
    
    output_files = []
    material_counts = defaultdict(int)
    
    for sheet in sheets:
        material_counts[sheet.material] += 1
        sheet_num = material_counts[sheet.material]
        
        # Clean material name for filename
        material_clean = sheet.material.replace(' ', '_').replace('/', '_')
        filename = f"{output_prefix}_{material_clean}_{sheet_num}.svg"

        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        generator = SVGGenerator(sheet)
        svg_content = generator.generate()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        output_files.append(filename)
        
        print(f"\n📄 {filename}")
        print(f"   Material: {sheet.material}")
        print(f"   Pieces: {len(sheet.placements)}")
        print(f"   Efficiency: {sheet.efficiency:.1f}%")
    
    print(f"\n{'═' * 70}")
    print(f"✅ SUCCESS: Generated {len(output_files)} SVG file(s)")
    print(f"{'═' * 70}")
    
    # Summary by material
    print("\nSUMMARY BY MATERIAL:")
    material_summary = defaultdict(lambda: {'sheets': 0, 'pieces': 0})
    for sheet in sheets:
        material_summary[sheet.material]['sheets'] += 1
        material_summary[sheet.material]['pieces'] += len(sheet.placements)
    
    for material, stats in sorted(material_summary.items()):
        print(f"  • {material}: {stats['sheets']} sheet(s), {stats['pieces']} piece(s)")
    
    print()
    
    return output_files


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate laser cut sheets from CSV using ILP optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generator.py INPUT.csv 300 200
  python generator.py INPUT.csv 800 600 -o project_name
  python generator.py INPUT.csv 300 200 --solver ffd           # fast heuristic
  python generator.py INPUT.csv 300 200 --time-limit 120       # 2-min ILP limit

Requires for ILP solver:  pip install ortools
Requires for SVG output:  pip install svgwrite fonttools

CSV Format:
  "QUANTITY";"MATERIAL";"TEXTDATA";"LABEL WIDTH(mm)";"LABEL HEIGHT(mm)";"TEXT HEIGHT(mm)"

Output SVG layers:
  borders   — amber (#ffc107) 0.8 mm etched frame
  cut_lines — red   (#f44336) cutting outline
  text      — blue  (#0000ff) text-to-path
        """
    )

    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument("width",  type=float, help="Sheet width in mm")
    parser.add_argument("height", type=float, help="Sheet height in mm")
    parser.add_argument("-o", "--output", default="output/cutsheet",
                        help="Output path prefix, e.g. output/cutsheet "
                             "(default: output/cutsheet). "
                             "The directory is created automatically.")
    parser.add_argument("--solver", choices=["ilp", "ffd"], default="ilp",
                        help="Solver: 'ilp' = OR-Tools CP-SAT (default), "
                             "'ffd' = First-Fit Decreasing heuristic")
    parser.add_argument("--time-limit", type=int, default=60,
                        metavar="SECONDS",
                        help="CP-SAT wall-clock time limit per material group "
                             "(default: 60s, ILP only)")

    args = parser.parse_args()

    try:
        generate_laser_sheets(args.csv_file, args.width, args.height,
                              args.output, args.solver, args.time_limit)
    except FileNotFoundError:
        print(f"\n❌ Error: File '{args.csv_file}' not found")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)