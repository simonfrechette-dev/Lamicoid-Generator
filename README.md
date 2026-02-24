# Lamicoid Generator

A Python tool that generates optimised SVG cut sheets for laser engraving and cutting lamicoid labels (engraved nameplates used in industrial and electrical panel applications).

Given a CSV list of label definitions, the tool solves a **2-D bin-packing problem** to minimise the number of stock sheets required, then renders each sheet as a ready-to-use SVG file with separate layers for cutting, border engraving, and text engraving.

---

## Features

- **Optimal or fast packing** — choose between an exact ILP solver (OR-Tools CP-SAT) or a quick First-Fit Decreasing heuristic
- **Multi-material support** — labels are grouped by material; each output sheet contains only one material type
- **Label rotation** — both 0° and 90° orientations are tried to improve packing density
- **Text to paths** — label text is converted to filled SVG path outlines via fontTools (no font embedding required in the output file)
- **Geometry optimisation** — adjacent border and cut-line rectangles are merged to reduce element count and improve laser throughput
- **Three named SVG layers** — compatible with laser software that reads layer/group colours for operation assignment

---

## Requirements

| Package | Purpose | Required? |
|---------|---------|-----------|
| `svgwrite` | SVG generation | Yes |
| `fonttools` | Text-to-path conversion | Yes |
| `ortools` | ILP/CP-SAT optimal solver | Optional (falls back to FFD) |

Python 3.8 or later is required.

---

## Installation

### Windows (step-by-step)

**1 — Install Python 3.8 or later**

Download the installer from <https://www.python.org/downloads/windows/>.  
During setup, tick **"Add Python to PATH"** before clicking *Install Now*.

Verify in a new Command Prompt:
```
python --version
```

---

**2 — Install Git** *(optional, for cloning)*

Download the installer from <https://git-scm.com/download/win> and run it.  
The default options are fine.  Verify in a new Command Prompt:
```
git --version
```

If you prefer not to install Git, skip this step and download the ZIP from GitHub instead (see step 3).

---

**3 — Download the project**

Either clone with Git:
```
git clone https://github.com/<your-username>/lamicoid-generator.git
cd "lamicoid-generator"
```
or download and extract the ZIP from GitHub, then open a Command Prompt in the extracted folder.

---

**4 — Create a virtual environment** *(recommended)*

```
python -m venv .venv
.venv\Scripts\activate
```

Your prompt will change to `(.venv) C:\…>` when the environment is active.  
Run `.venv\Scripts\activate` again any time you open a new Command Prompt.

---

**5 — Install dependencies**

```
pip install svgwrite fonttools
```

Optional — for the optimal ILP solver (requires ~500 MB disk space):
```
pip install ortools
```

---

**6 — Font** *(optional)*

Arial is built into Windows and is found automatically at `C:\Windows\Fonts\arial.ttf`.  
No extra steps are needed.

---

**7 — Test the installation**

```
python generator.py INPUT.csv 300 200 --solver ffd
```

You should see SVG files created in the current folder without any error messages.

---

**Deactivate the environment** when you are done:
```
deactivate
```

---

### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/<your-username>/lamicoid-generator.git
cd lamicoid-generator

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install svgwrite fonttools

# Optional: install OR-Tools for optimal bin packing
pip install ortools
```

### Font

The generator looks for a TrueType font in the following locations (in order):

1. `arial.ttf` in the current working directory
2. `/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf` (Linux)
3. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` (Linux)
4. `/System/Library/Fonts/Helvetica.ttc` (macOS)
5. `C:\Windows\Fonts\arial.ttf` (Windows)

If no font is found, text is rendered as placeholder rectangles.

---

## Usage

```
python generator.py INPUT.csv SHEET_WIDTH SHEET_HEIGHT [options]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `INPUT.csv` | Path to the label definitions CSV |
| `SHEET_WIDTH` | Stock sheet width in mm |
| `SHEET_HEIGHT` | Stock sheet height in mm |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output PREFIX` | `output/cutsheet` | Output path prefix, including optional directory (created automatically) |
| `--solver {ilp,ffd}` | `ilp` | Solver: `ilp` = OR-Tools CP-SAT (optimal), `ffd` = First-Fit Decreasing (fast) |
| `--time-limit SECONDS` | `60` | CP-SAT time limit per material group (ILP only) |

### Examples

```bash
# Basic usage — 300 × 200 mm sheets, ILP solver
python generator.py INPUT.csv 300 200

# Custom output directory and prefix
python generator.py INPUT.csv 800 600 -o my_project/cutsheet

# Fast heuristic (no OR-Tools required)
python generator.py INPUT.csv 300 200 --solver ffd

# Allow 2 minutes for the ILP solver per material group
python generator.py INPUT.csv 300 200 --time-limit 120
```

Output SVG files are named `<prefix>_<material>_<sheet_number>.svg`, e.g. `output/cutsheet_ABS_WHITE_1.svg`.  The output directory is created automatically if it does not exist.

---

## CSV Format

The input file must have a **header row** and **UTF-8 encoding**.  The delimiter (`,`, `;`, `\t`, or `|`) and quote character are detected automatically.  Semicolon-delimited files with double-quote quoting (the default export format of most European spreadsheet applications) are fully supported.

```
"QUANTITY";"MATERIAL";"TEXTDATA";"LABEL WIDTH(mm)";"LABEL HEIGHT(mm)";"TEXT HEIGHT(mm)"
14;"ABS WHITE";"POWER SUPPLY\n120VAC";88.9;38.1;6
5;"ABS YELLOW";"CAUTION\nDO NOT BYPASS";88.9;38.1;6
```

### Column reference

| Column | Accepted aliases | Description |
|--------|-----------------|-------------|
| `QUANTITY` | `QTY`, `COUNT` | Number of identical copies |
| `MATERIAL` | `MAT`, `MATERIAL TYPE` | Stock material identifier (used to group labels onto sheets) |
| `TEXTDATA` | `TEXT`, `LABEL TEXT`, `CONTENT` | Engraved text; use `\n` for line breaks |
| `LABEL WIDTH(mm)` | `WIDTH(MM)`, `WIDTH`, `LABEL WIDTH` | Label width in mm |
| `LABEL HEIGHT(mm)` | `HEIGHT(MM)`, `HEIGHT`, `LABEL HEIGHT` | Label height in mm |
| `TEXT HEIGHT(mm)` | `TEXT SIZE(MM)`, `TEXT HEIGHT`, `FONT SIZE` | Cap-height of the text in mm |

Multi-line text values are supported as CSV-quoted fields with embedded newlines or with `\n` in the text content.

---

## Output SVG Layers

Each generated SVG contains three named groups, colour-coded for laser machine operation assignment:

| Group id | Colour | Hex | Purpose |
|----------|--------|-----|---------|
| `borders` | Amber | `#ffc107` | 0.8 mm filled frame around each label — laser **engraves** the border |
| `cut_lines` | Red | `#f44336` | 0.1 mm stroke outline at each label boundary — laser **cuts** here |
| `text` | Blue | `#0000ff` | Label text converted to filled path outlines — laser **engraves** the text |

---

## Algorithm Overview

### Bin Packing

**ILP solver (default)** — `ILPBinPacker`

Uses [OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model) to find a provably optimal placement that minimises the number of sheets. The formulation uses optional interval variables and `AddNoOverlap2D` constraints. The FFD heuristic result is used as an upper bound and as a warm-start hint before solving.

**FFD heuristic** — `SimpleBinPacker`

First-Fit Decreasing with a skyline placement strategy. Labels are sorted largest-area-first and placed into the first feasible position found. No external dependencies required. Used as fallback when OR-Tools is not installed.

Both solvers try both 0° and 90° rotations for every label.

### SVG Geometry

After placement, adjacent border and cut-line rectangles are merged (end-to-end, side-by-side, and overlapping strategies) over iterative passes until no further merges are possible. This reduces the total path count and can improve laser throughput on machines that process paths sequentially.

Text is rendered by walking each character through the font's glyph set via fontTools, applying a scale + y-flip affine transform to convert from the font's y-up coordinate system to SVG's y-down system. All glyphs for a label are accumulated into a single merged path string.

---

## Project Structure

```
lamicoid-generator/
├── generator.py   # Main script — data structures, solvers, SVG renderer, CLI
├── INPUT.csv      # Example label definitions
└── arial.ttf      # Optional: place a TrueType font here for text rendering
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
