#!/usr/bin/env python3
# Copyright (C) 2026  Lesco Design & Mfg. Co., Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Test suite for generator.py
============================
Covers:
  - Data models  : Label, Placement, Sheet
  - SimpleBinPacker (FFD): packing logic, skyline, overlap checks
  - ILPBinPacker (CP-SAT): optimal packing and FFD fallback
  - parse_csv()          : dialect detection, multi-line text, error handling
  - SVGGenerator         : SVG structure, layer content, geometry merging
  - generate_laser_sheets: end-to-end integration (FFD & ILP)

Run with:
    py -m pytest test_generator.py -v
  or:
    py test_generator.py
"""

import configparser
import csv
import io
import os
import re
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Import module under test (suppress config-load print)
# ---------------------------------------------------------------------------
import contextlib

with contextlib.redirect_stdout(io.StringIO()):
    from generator import (
        CFG,
        ILPBinPacker,
        Label,
        Placement,
        Sheet,
        SimpleBinPacker,
        SVGGenerator,
        generate_laser_sheets,
        load_config,
        parse_csv,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(w=50.0, h=30.0, mat="MAT", text="TXT", th=5.0, qty=1) -> Label:
    return Label(material=mat, width=w, height=h, text=text,
                 text_height=th, quantity=qty)


def _make_sheet(w=300.0, h=200.0, placements=None, mat="MAT") -> Sheet:
    return Sheet(width=w, height=h, placements=placements or [], material=mat)


def _minimal_cfg() -> configparser.ConfigParser:
    """Return a ConfigParser with all required keys pre-populated."""
    cfg = configparser.ConfigParser(defaults={
        'cut': '#f44336', 'engrave': '#ffc107', 'text': '#0000ff',
        'border_width': '0.8', 'text_margin': '1.0',
        'width': '300', 'height': '200', 'name': 'Arial',
    })
    for section in ('colors', 'dimensions', 'sheet', 'font'):
        cfg.add_section(section)
    return cfg


def _write_csv(rows: list[str], tmp_dir: str, filename: str = "test.csv") -> str:
    path = os.path.join(tmp_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))
    return path


# ============================================================================
# 1. Data-model tests
# ============================================================================

class TestLabel(unittest.TestCase):

    def test_area_square(self):
        self.assertAlmostEqual(Label("M", 10, 10, "", 5).area, 100.0)

    def test_area_rectangle(self):
        self.assertAlmostEqual(_make_label(w=88.9, h=38.1).area, 88.9 * 38.1)

    def test_area_positive(self):
        self.assertGreater(_make_label().area, 0)

    def test_default_quantity(self):
        self.assertEqual(_make_label().quantity, 1)


class TestSheet(unittest.TestCase):

    def test_efficiency_empty_sheet(self):
        sheet = _make_sheet()
        self.assertAlmostEqual(sheet.efficiency, 0.0)

    def test_efficiency_full_sheet(self):
        label = _make_label(w=300.0, h=200.0)
        sheet = _make_sheet(w=300.0, h=200.0,
                            placements=[Placement(label, 0, 0)])
        self.assertAlmostEqual(sheet.efficiency, 100.0)

    def test_efficiency_partial(self):
        # One 50×30 label on a 300×200 sheet = 1500/60000 = 2.5%
        label = _make_label(w=50.0, h=30.0)
        sheet = _make_sheet(w=300.0, h=200.0,
                            placements=[Placement(label, 0, 0)])
        expected = (50.0 * 30.0) / (300.0 * 200.0) * 100
        self.assertAlmostEqual(sheet.efficiency, expected)

    def test_efficiency_zero_area_sheet(self):
        sheet = _make_sheet(w=0, h=0)
        self.assertAlmostEqual(sheet.efficiency, 0.0)


# ============================================================================
# 2. SimpleBinPacker tests
# ============================================================================

class TestSimpleBinPackerOverlap(unittest.TestCase):

    def setUp(self):
        self.packer = SimpleBinPacker(300, 200)

    def _sheet_with(self, *placements):
        return _make_sheet(placements=list(placements))

    def test_no_overlap_side_by_side(self):
        label = _make_label(w=50, h=30)
        p = Placement(label, 0, 0)
        sheet = self._sheet_with(p)
        # New position is to the right → no overlap
        self.assertFalse(self.packer._check_overlap(sheet, 50, 0, 50, 30))

    def test_no_overlap_above_below(self):
        label = _make_label(w=50, h=30)
        p = Placement(label, 0, 0)
        sheet = self._sheet_with(p)
        self.assertFalse(self.packer._check_overlap(sheet, 0, 30, 50, 30))

    def test_overlap_full_cover(self):
        label = _make_label(w=50, h=30)
        p = Placement(label, 0, 0)
        sheet = self._sheet_with(p)
        self.assertTrue(self.packer._check_overlap(sheet, 0, 0, 50, 30))

    def test_overlap_partial(self):
        label = _make_label(w=50, h=30)
        p = Placement(label, 0, 0)
        sheet = self._sheet_with(p)
        # Shift 25 units right → overlaps by 25 mm
        self.assertTrue(self.packer._check_overlap(sheet, 25, 0, 50, 30))

    def test_no_overlap_rotated_placement(self):
        label = _make_label(w=50, h=30)
        # Rotated 90°: effective w=30, h=50
        p = Placement(label, 0, 0, rotation=90)
        sheet = self._sheet_with(p)
        # Should occupy x:[0,30] y:[0,50] — placing at x=30 should not overlap
        self.assertFalse(self.packer._check_overlap(sheet, 30, 0, 50, 30))


class TestSimpleBinPackerSkyline(unittest.TestCase):

    def setUp(self):
        self.packer = SimpleBinPacker(300, 200)

    def test_skyline_empty_sheet(self):
        sheet = _make_sheet()
        skyline = self.packer._build_skyline(sheet)
        self.assertEqual(skyline, [(0.0, 0.0)])

    def test_skyline_one_placement(self):
        label = _make_label(w=50, h=30)
        sheet = _make_sheet(placements=[Placement(label, 0, 0)])
        skyline = self.packer._build_skyline(sheet)
        # Should include origin and x=50
        self.assertIn((0.0, 0.0), skyline)
        self.assertIn((50.0, 0.0), skyline)

    def test_get_max_y_empty_sheet(self):
        sheet = _make_sheet()
        self.assertEqual(self.packer._get_max_y_at_position(sheet, 0, 50), 0.0)

    def test_get_max_y_with_placement(self):
        label = _make_label(w=50, h=30)
        sheet = _make_sheet(placements=[Placement(label, 0, 0)])
        # Band x=[0,50] overlaps placement → max_y = 30
        self.assertAlmostEqual(
            self.packer._get_max_y_at_position(sheet, 0, 50), 30.0)

    def test_get_max_y_non_overlapping_band(self):
        label = _make_label(w=50, h=30)
        sheet = _make_sheet(placements=[Placement(label, 0, 0)])
        # Band x=[100,150] does not overlap placement → max_y = 0
        self.assertAlmostEqual(
            self.packer._get_max_y_at_position(sheet, 100, 50), 0.0)


class TestSimpleBinPackerPack(unittest.TestCase):

    def setUp(self):
        self.packer = SimpleBinPacker(300, 200)

    def test_single_label_one_sheet(self):
        labels = [_make_label(w=50, h=30, qty=1)]
        sheets = self.packer.pack(labels)
        self.assertEqual(len(sheets), 1)
        self.assertEqual(len(sheets[0].placements), 1)

    def test_multiple_labels_fit_one_sheet(self):
        # 5 labels of 50×30 on a 300×200 sheet → all fit if packed well
        labels = [_make_label(w=50, h=30, qty=5)]
        sheets = self.packer.pack(labels)
        total = sum(len(s.placements) for s in sheets)
        self.assertEqual(total, 5)

    def test_overflow_to_second_sheet(self):
        # 50 labels 50×30 on 300×200 → some overflow
        labels = [_make_label(w=50, h=30, qty=50)]
        sheets = self.packer.pack(labels)
        self.assertGreater(len(sheets), 1)
        total = sum(len(s.placements) for s in sheets)
        self.assertEqual(total, 50)

    def test_material_grouping(self):
        labels = [
            _make_label(w=50, h=30, mat="RED", qty=2),
            _make_label(w=50, h=30, mat="BLUE", qty=2),
        ]
        sheets = self.packer.pack(labels)
        materials = {s.material for s in sheets}
        self.assertIn("RED", materials)
        self.assertIn("BLUE", materials)
        for sheet in sheets:
            self.assertEqual(
                len({p.label.material for p in sheet.placements}), 1,
                "All placements on a sheet must share one material")

    def test_rotation_used_when_needed(self):
        # A 180×10 label on 200×100 sheet fits only in landscape (original) or
        # portrait (rotated to 10×180). Either way, must be placed.
        small_packer = SimpleBinPacker(200, 100)
        labels = [_make_label(w=180, h=10, qty=1)]
        sheets = small_packer.pack(labels)
        total = sum(len(s.placements) for s in sheets)
        self.assertEqual(total, 1)

    def test_label_too_large_skipped_gracefully(self):
        # 400×300 label on 300×200 sheet — cannot fit at all
        labels = [_make_label(w=400, h=300, qty=1)]
        sheets = self.packer.pack(labels)
        # Generator opens a sheet, finds no placement, but doesn't crash
        total = sum(len(s.placements) for s in sheets)
        self.assertEqual(total, 0)

    def test_all_placements_within_sheet_bounds(self):
        labels = [_make_label(w=88.9, h=38.1, qty=14)]
        sheets = self.packer.pack(labels)
        for sheet in sheets:
            for p in sheet.placements:
                eff_w = p.label.width  if p.rotation == 0 else p.label.height
                eff_h = p.label.height if p.rotation == 0 else p.label.width
                self.assertGreaterEqual(p.x, 0,          "x must be >= 0")
                self.assertGreaterEqual(p.y, 0,          "y must be >= 0")
                self.assertLessEqual(p.x + eff_w, sheet.width  + 1e-6, "x+w overflows sheet")
                self.assertLessEqual(p.y + eff_h, sheet.height + 1e-6, "y+h overflows sheet")

    def test_no_placements_overlap(self):
        labels = [_make_label(w=50, h=30, qty=20)]
        sheets = self.packer.pack(labels)
        for sheet in sheets:
            for i, pi in enumerate(sheet.placements):
                wi = pi.label.width  if pi.rotation == 0 else pi.label.height
                hi = pi.label.height if pi.rotation == 0 else pi.label.width
                for j, pj in enumerate(sheet.placements):
                    if i >= j:
                        continue
                    wj = pj.label.width  if pj.rotation == 0 else pj.label.height
                    hj = pj.label.height if pj.rotation == 0 else pj.label.width
                    # AABB non-overlap check
                    overlap = not (
                        pi.x + wi <= pj.x or pj.x + wj <= pi.x or
                        pi.y + hi <= pj.y or pj.y + hj <= pi.y
                    )
                    self.assertFalse(
                        overlap,
                        f"Placements {i} and {j} overlap on sheet")


# ============================================================================
# 3. ILPBinPacker tests
# ============================================================================

class TestILPBinPackerFallback(unittest.TestCase):
    """Tests that run regardless of whether ortools is installed."""

    def test_pack_returns_all_labels(self):
        packer = ILPBinPacker(300, 200, time_limit=10)
        labels = [_make_label(w=50, h=30, qty=3)]
        with contextlib.redirect_stdout(io.StringIO()):
            sheets = packer.pack(labels)
        total = sum(len(s.placements) for s in sheets)
        self.assertEqual(total, 3)

    def test_material_grouping_preserved(self):
        packer = ILPBinPacker(300, 200, time_limit=10)
        labels = [
            _make_label(w=50, h=30, mat="A", qty=2),
            _make_label(w=50, h=30, mat="B", qty=2),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            sheets = packer.pack(labels)
        for sheet in sheets:
            self.assertEqual(
                len({p.label.material for p in sheet.placements}), 1)

    def test_ilp_uses_fewer_or_equal_sheets_than_ffd(self):
        """ILP must never produce *more* sheets than FFD."""
        labels = [_make_label(w=80, h=40, qty=12)]
        ffd = SimpleBinPacker(300, 200)
        ilp = ILPBinPacker(300, 200, time_limit=15)
        with contextlib.redirect_stdout(io.StringIO()):
            ffd_sheets = ffd.pack(labels)
            ilp_sheets = ilp.pack(labels)
        self.assertLessEqual(len(ilp_sheets), len(ffd_sheets))

    def test_no_overlaps_in_ilp_result(self):
        labels = [_make_label(w=60, h=40, qty=8)]
        packer = ILPBinPacker(300, 200, time_limit=15)
        with contextlib.redirect_stdout(io.StringIO()):
            sheets = packer.pack(labels)
        for sheet in sheets:
            for i, pi in enumerate(sheet.placements):
                wi = pi.label.width  if pi.rotation == 0 else pi.label.height
                hi = pi.label.height if pi.rotation == 0 else pi.label.width
                for j, pj in enumerate(sheet.placements):
                    if i >= j:
                        continue
                    wj = pj.label.width  if pj.rotation == 0 else pj.label.height
                    hj = pj.label.height if pj.rotation == 0 else pj.label.width
                    overlap = not (
                        pi.x + wi <= pj.x or pj.x + wj <= pi.x or
                        pi.y + hi <= pj.y or pj.y + hj <= pi.y
                    )
                    self.assertFalse(overlap, f"ILP placements {i} and {j} overlap")


# ============================================================================
# 4. parse_csv tests
# ============================================================================

HEADER = '"QUANTITY";"MATERIAL";"TEXTDATA";"LABEL WIDTH(mm)";"LABEL HEIGHT(mm)";"TEXT HEIGHT(mm)"'


class TestParseCSV(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _parse(self, rows):
        path = _write_csv(rows, self.tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            return parse_csv(path)

    # ── basic parsing ──────────────────────────────────────────────────

    def test_basic_row(self):
        labels = self._parse([HEADER, '2;"ABS WHITE";"PUMP A";50;30;5'])
        self.assertEqual(len(labels), 1)
        lbl = labels[0]
        self.assertEqual(lbl.quantity, 2)
        self.assertEqual(lbl.material, "ABS WHITE")
        self.assertEqual(lbl.text, "PUMP A")
        self.assertAlmostEqual(lbl.width, 50.0)
        self.assertAlmostEqual(lbl.height, 30.0)
        self.assertAlmostEqual(lbl.text_height, 5.0)

    def test_multiple_rows(self):
        labels = self._parse([
            HEADER,
            '1;"M1";"A";50;30;5',
            '3;"M2";"B";60;40;6',
        ])
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].quantity, 1)
        self.assertEqual(labels[1].quantity, 3)

    # ── multi-line text ───────────────────────────────────────────────

    def test_multiline_text_embedded_newline(self):
        # Actual newline inside a quoted field
        rows = [
            HEADER,
            '1;"M";"LINE1\nLINE2";50;30;5',
        ]
        labels = self._parse(rows)
        self.assertIn('\n', labels[0].text)
        self.assertEqual(labels[0].text.count('\n'), 1)

    def test_multiline_text_escaped_backslash_n(self):
        # Literal backslash-n in the CSV value
        path = _write_csv([HEADER, r'1;"M";"LINE1\nLINE2";50;30;5'], self.tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            labels = parse_csv(path)
        self.assertIn('\n', labels[0].text)

    # ── error / edge cases ────────────────────────────────────────────

    def test_empty_file(self):
        labels = self._parse([])
        self.assertEqual(labels, [])

    def test_header_only(self):
        labels = self._parse([HEADER])
        self.assertEqual(labels, [])

    def test_missing_required_field(self):
        bad_header = '"QTY";"MATERIAL";"TEXTDATA";"LABEL WIDTH(mm)";"LABEL HEIGHT(mm)"'
        labels = self._parse([bad_header, '1;"M";"T";50;30'])
        self.assertEqual(labels, [])

    def test_invalid_numeric_row_skipped(self):
        labels = self._parse([
            HEADER,
            'NOTANUMBER;"M";"T";50;30;5',  # invalid quantity → skipped
            '1;"M";"T";50;30;5',           # valid
        ])
        self.assertEqual(len(labels), 1)

    def test_empty_rows_skipped(self):
        labels = self._parse([HEADER, '', '1;"M";"T";50;30;5', ''])
        self.assertEqual(len(labels), 1)

    # ── real INPUT.csv ────────────────────────────────────────────────

    def test_real_input_csv(self):
        real_csv = os.path.join(
            os.path.dirname(__file__), 'INPUT.csv')
        if not os.path.exists(real_csv):
            self.skipTest("INPUT.csv not found in workspace")
        with contextlib.redirect_stdout(io.StringIO()):
            labels = parse_csv(real_csv)
        self.assertGreater(len(labels), 0)
        for lbl in labels:
            self.assertGreater(lbl.quantity, 0)
            self.assertGreater(lbl.width, 0)
            self.assertGreater(lbl.height, 0)
            self.assertGreater(lbl.text_height, 0)


# ============================================================================
# 5. SVGGenerator tests
# ============================================================================

def _make_svg_generator(w=88.9, h=38.1, text="PUMP A", qty=1,
                        rotation=0) -> SVGGenerator:
    label = _make_label(w=w, h=h, text=text, th=5.0, qty=qty)
    placement = Placement(label, 0.0, 0.0, rotation=rotation)
    sheet = _make_sheet(w=300, h=200, placements=[placement])
    with contextlib.redirect_stdout(io.StringIO()):
        return SVGGenerator(sheet, _minimal_cfg())


class TestSVGGeneratorStructure(unittest.TestCase):

    def _get_svg(self, **kwargs) -> str:
        gen = _make_svg_generator(**kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return gen.generate()

    def test_svg_element_present(self):
        svg = self._get_svg()
        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)

    def test_xml_declaration_present(self):
        svg = self._get_svg()
        self.assertTrue(svg.lstrip().startswith('<?xml') or svg.lstrip().startswith('<!--'),
                        "SVG should start with XML declaration or metadata comment")

    def test_all_three_layers_present(self):
        svg = self._get_svg()
        self.assertIn('id="cut_lines"', svg)
        self.assertIn('id="borders"',   svg)
        self.assertIn('id="text"',      svg)

    def test_correct_colours_in_styles(self):
        svg = self._get_svg()
        self.assertIn('#f44336', svg)   # cut
        self.assertIn('#ffc107', svg)   # engrave
        self.assertIn('#0000ff', svg)   # text

    def test_viewbox_matches_sheet(self):
        svg = self._get_svg()
        self.assertIn('viewBox="0 0 300 200"', svg)

    def test_metadata_comment(self):
        svg = self._get_svg()
        self.assertIn('Laser Cut Sheet Generator', svg)
        self.assertIn('Material:', svg)
        self.assertIn('Efficiency:', svg)

    def test_rotated_label_produces_svg(self):
        svg = self._get_svg(rotation=90)
        self.assertIn('<svg', svg)
        self.assertIn('id="cut_lines"', svg)

    def test_empty_text_no_path(self):
        label = _make_label(w=50, h=30, text="")
        sheet = _make_sheet(placements=[Placement(label, 0, 0)])
        with contextlib.redirect_stdout(io.StringIO()):
            gen = SVGGenerator(sheet, _minimal_cfg())
            svg = gen.generate()
        # text group should be empty (no path element inside it)
        text_group_match = re.search(
            r'id="text"[^>]*>(.*?)</g>', svg, re.DOTALL)
        if text_group_match:
            group_content = text_group_match.group(1).strip()
            self.assertNotIn('<path', group_content)


class TestSVGGeneratorExtractPlacement(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()

    def test_returns_three_lists(self):
        label   = _make_label(w=50, h=30, text="X")
        p       = Placement(label, 10, 20, 0)
        borders, cuts, paths = self.gen._extract_placement_elements(p, 0)
        self.assertIsInstance(borders, list)
        self.assertIsInstance(cuts, list)
        self.assertIsInstance(paths, list)

    def test_four_border_rects(self):
        label   = _make_label(w=50, h=30)
        p       = Placement(label, 0, 0, 0)
        borders, _, _ = self.gen._extract_placement_elements(p, 0)
        self.assertEqual(len(borders), 4)

    def test_four_cut_lines(self):
        label = _make_label(w=50, h=30)
        p = Placement(label, 0, 0, 0)
        _, cuts, _ = self.gen._extract_placement_elements(p, 0)
        self.assertEqual(len(cuts), 4)

    def test_cut_lines_have_correct_keys(self):
        label = _make_label(w=50, h=30)
        p = Placement(label, 10, 20, 0)
        _, cuts, _ = self.gen._extract_placement_elements(p, 0)
        for cut in cuts:
            for key in ('x1', 'y1', 'x2', 'y2', 'orientation'):
                self.assertIn(key, cut)

    def test_rotation_swaps_dimensions(self):
        label = _make_label(w=80, h=20)
        p0  = Placement(label, 0, 0, 0)
        p90 = Placement(label, 0, 0, 90)
        b0,  _, _ = self.gen._extract_placement_elements(p0,  0)
        b90, _, _ = self.gen._extract_placement_elements(p90, 0)
        # Top border: width should be label.width(0°) vs label.height(90°)
        w0  = next(r for r in b0  if r['y'] == 0)['width']
        w90 = next(r for r in b90 if r['y'] == 0)['width']
        self.assertAlmostEqual(w0,  80.0)
        self.assertAlmostEqual(w90, 20.0)

    def test_text_path_present_when_text_non_empty(self):
        label = _make_label(w=88.9, h=38.1, text="HELLO")
        p = Placement(label, 0, 0, 0)
        _, _, paths = self.gen._extract_placement_elements(p, 0)
        self.assertGreater(len(paths), 0)
        self.assertIn('d', paths[0])

    def test_no_text_path_when_text_empty(self):
        label = _make_label(w=50, h=30, text="")
        p = Placement(label, 0, 0, 0)
        _, _, paths = self.gen._extract_placement_elements(p, 0)
        self.assertEqual(paths, [])

    def test_rotated_text_gets_transform(self):
        label = _make_label(w=88.9, h=38.1, text="HELLO")
        p = Placement(label, 0, 0, 90)
        _, _, paths = self.gen._extract_placement_elements(p, 0)
        if paths:
            self.assertIn('rotate', paths[0].get('transform', ''))


# ============================================================================
# 6. Geometry-optimisation tests
# ============================================================================

class TestOptimizeRectangles(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()

    # helpers
    def _h(self, x, y, w, h):
        return {'x': x, 'y': y, 'width': w, 'height': h, 'orientation': 'horizontal'}

    def _v(self, x, y, w, h):
        return {'x': x, 'y': y, 'width': w, 'height': h, 'orientation': 'vertical'}

    def test_empty_list(self):
        self.assertEqual(self.gen._optimize_rectangles([]), [])

    def test_single_rect_unchanged(self):
        r = [self._h(0, 0, 100, 0.8)]
        result = self.gen._optimize_rectangles(r)
        self.assertEqual(len(result), 1)

    def test_end_to_end_horizontal_merge(self):
        r = [self._h(0, 0, 50, 0.8), self._h(50, 0, 50, 0.8)]
        result = self.gen._optimize_rectangles(r)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['width'], 100.0)

    def test_end_to_end_vertical_merge(self):
        r = [self._v(0, 0, 0.8, 40), self._v(0, 40, 0.8, 40)]
        result = self.gen._optimize_rectangles(r)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['height'], 80.0)

    def test_non_adjacent_horizontal_not_merged(self):
        r = [self._h(0, 0, 40, 0.8), self._h(50, 0, 40, 0.8)]  # gap of 10
        result = self.gen._optimize_rectangles(r)
        self.assertEqual(len(result), 2)

    def test_different_y_horizontal_not_merged(self):
        r = [self._h(0, 0, 50, 0.8), self._h(50, 5, 50, 0.8)]  # different Y
        result = self.gen._optimize_rectangles(r)
        self.assertEqual(len(result), 2)


class TestTryMergeEndToEnd(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()
        self.TOL = 0.01

    def _h(self, x, y, w, h):
        return {'x': x, 'y': y, 'width': w, 'height': h, 'orientation': 'horizontal'}

    def _v(self, x, y, w, h):
        return {'x': x, 'y': y, 'width': w, 'height': h, 'orientation': 'vertical'}

    def test_horizontal_adjacent_right(self):
        r1 = self._h(0, 0, 50, 2)
        r2 = self._h(50, 0, 50, 2)
        result = self.gen._try_merge_end_to_end(r1, r2, self.TOL)
        self.assertIsNotNone(result, "Expected a merged result but got None")
        if result is not None:
            self.assertAlmostEqual(result['width'], 100.0)

    def test_horizontal_adjacent_left(self):
        r1 = self._h(50, 0, 50, 2)
        r2 = self._h(0, 0, 50, 2)
        result = self.gen._try_merge_end_to_end(r1, r2, self.TOL)
        self.assertIsNotNone(result, "Expected a merged result but got None")
        if result is not None:
            self.assertAlmostEqual(result['x'], 0.0)
            self.assertAlmostEqual(result['width'], 100.0)

    def test_different_orientations_returns_none(self):
        r1 = self._h(0, 0, 50, 2)
        r2 = self._v(50, 0, 2, 50)
        self.assertIsNone(self.gen._try_merge_end_to_end(r1, r2, self.TOL))

    def test_vertical_adjacent_below(self):
        r1 = self._v(0, 0, 2, 50)
        r2 = self._v(0, 50, 2, 50)
        result = self.gen._try_merge_end_to_end(r1, r2, self.TOL)
        self.assertIsNotNone(result, "Expected a merged result but got None")
        if result is not None:
            self.assertAlmostEqual(result['height'], 100.0)

    def test_gap_returns_none(self):
        r1 = self._h(0, 0, 40, 2)
        r2 = self._h(50, 0, 40, 2)   # 10-unit gap
        self.assertIsNone(self.gen._try_merge_end_to_end(r1, r2, self.TOL))


class TestTryMergeSideBySide(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()
        self.TOL = 0.01

    def _h(self, x, y, w, h):
        return {'x': x, 'y': y, 'width': w, 'height': h, 'orientation': 'horizontal'}

    def _v(self, x, y, w, h):
        return {'x': x, 'y': y, 'width': w, 'height': h, 'orientation': 'vertical'}

    def test_horizontal_stacked_merge(self):
        r1 = self._h(0, 0, 100, 2)
        r2 = self._h(0, 2, 100, 2)
        result = self.gen._try_merge_side_by_side(r1, r2, self.TOL)
        self.assertIsNotNone(result, "Expected a merged result but got None")
        if result is not None:
            self.assertAlmostEqual(result['height'], 4.0)

    def test_horizontal_different_x_returns_none(self):
        r1 = self._h(0, 0, 100, 2)
        r2 = self._h(5, 2, 100, 2)
        self.assertIsNone(self.gen._try_merge_side_by_side(r1, r2, self.TOL))

    def test_vertical_side_by_side_merge(self):
        r1 = self._v(0, 0, 2, 100)
        r2 = self._v(2, 0, 2, 100)
        result = self.gen._try_merge_side_by_side(r1, r2, self.TOL)
        self.assertIsNotNone(result, "Expected a merged result but got None")
        if result is not None:
            self.assertAlmostEqual(result['width'], 4.0)
            self.assertAlmostEqual(result['height'], 200.0)


class TestOptimizeCutLines(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()

    def _h(self, x1, y, x2):
        return {'x1': x1, 'y1': y, 'x2': x2, 'y2': y, 'orientation': 'horizontal'}

    def _v(self, x, y1, y2):
        return {'x1': x, 'y1': y1, 'x2': x, 'y2': y2, 'orientation': 'vertical'}

    def test_empty_returns_empty(self):
        self.assertEqual(self.gen._optimize_cut_lines([]), [])

    def test_single_line_unchanged(self):
        lines = [self._h(0, 0, 50)]
        self.assertEqual(len(self.gen._optimize_cut_lines(lines)), 1)

    def test_collinear_horizontal_adjacent_merge(self):
        lines = [self._h(0, 0, 50), self._h(50, 0, 100)]
        result = self.gen._optimize_cut_lines(lines)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(min(result[0]['x1'], result[0]['x2']), 0.0)
        self.assertAlmostEqual(max(result[0]['x1'], result[0]['x2']), 100.0)

    def test_collinear_vertical_adjacent_merge(self):
        lines = [self._v(0, 0, 50), self._v(0, 50, 100)]
        result = self.gen._optimize_cut_lines(lines)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(max(result[0]['y1'], result[0]['y2']), 100.0)

    def test_different_y_horizontal_not_merged(self):
        lines = [self._h(0, 0, 50), self._h(0, 10, 50)]
        result = self.gen._optimize_cut_lines(lines)
        self.assertEqual(len(result), 2)

    def test_different_x_vertical_not_merged(self):
        lines = [self._v(0, 0, 50), self._v(10, 0, 50)]
        result = self.gen._optimize_cut_lines(lines)
        self.assertEqual(len(result), 2)

    def test_overlapping_on_same_axis_merged(self):
        lines = [self._h(0, 0, 60), self._h(40, 0, 100)]  # overlap 40..60
        result = self.gen._optimize_cut_lines(lines)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(max(result[0]['x1'], result[0]['x2']), 100.0)


# ============================================================================
# 7. _text_to_paths and _measure_text_line_width tests
# ============================================================================

class TestTextPaths(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()

    def test_empty_text_returns_empty_list(self):
        result = self.gen._text_to_paths("", 0, 0, 50, 30, 5)
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty(self):
        result = self.gen._text_to_paths("   ", 0, 0, 50, 30, 5)
        self.assertEqual(result, [])

    def test_single_line_returns_one_path(self):
        result = self.gen._text_to_paths("PUMP A", 0, 0, 88.9, 38.1, 5)
        self.assertEqual(len(result), 1)
        self.assertIn('d', result[0])

    def test_multiline_single_path(self):
        result = self.gen._text_to_paths("LINE1\nLINE2", 0, 0, 88.9, 38.1, 5)
        self.assertEqual(len(result), 1)

    def test_too_small_label_returns_empty(self):
        # A 2×2 mm label cannot accommodate any border+margin+text
        result = self.gen._text_to_paths("X", 0, 0, 2, 2, 5)
        self.assertEqual(result, [])

    def test_text_height_auto_shrunk_for_tall_block(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = self.gen._text_to_paths(
                "A\nB\nC\nD\nE", 0, 0, 88.9, 15.0, 5.0
            )
        output = out.getvalue()
        # Either the text is shrunk (warning printed) or returns empty
        if result:
            self.assertIn('d', result[0])


class TestMeasureTextLineWidth(unittest.TestCase):

    def setUp(self):
        self.gen = _make_svg_generator()

    def test_empty_string_is_zero(self):
        self.assertAlmostEqual(
            self.gen._measure_text_line_width("", 5.0), 0.0)

    def test_width_positive_for_non_empty(self):
        w = self.gen._measure_text_line_width("HELLO", 5.0)
        self.assertGreater(w, 0.0)

    def test_longer_text_is_wider(self):
        w_short = self.gen._measure_text_line_width("AB",  5.0)
        w_long  = self.gen._measure_text_line_width("ABCDE", 5.0)
        self.assertGreater(w_long, w_short)

    def test_larger_size_is_wider(self):
        w_small = self.gen._measure_text_line_width("HELLO", 3.0)
        w_large = self.gen._measure_text_line_width("HELLO", 8.0)
        self.assertGreater(w_large, w_small)


# ============================================================================
# 8. load_config tests
# ============================================================================

class TestLoadConfig(unittest.TestCase):

    def test_defaults_when_no_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config("/nonexistent/path/generator.conf")
        self.assertEqual(cfg.get('DEFAULT', 'cut'), '#f44336')
        self.assertEqual(cfg.get('DEFAULT', 'engrave'), '#ffc107')
        self.assertEqual(cfg.get('DEFAULT', 'text'), '#0000ff')
        self.assertAlmostEqual(cfg.getfloat('DEFAULT', 'border_width'), 0.8)

    def test_all_required_sections_created(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config("/nonexistent/path/generator.conf")
        for section in ('colors', 'dimensions', 'sheet', 'font'):
            self.assertTrue(cfg.has_section(section))

    def test_real_conf_file_loads(self):
        real_conf = os.path.join(os.path.dirname(__file__), 'generator.conf')
        if not os.path.exists(real_conf):
            self.skipTest("generator.conf not found")
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(real_conf)
        # Config file loaded: sections present
        self.assertTrue(cfg.has_section('colors') or cfg.has_section('sheet'))


# ============================================================================
# 9. End-to-end integration tests
# ============================================================================

_E2E_CSV_ROWS = [
    HEADER,
    '4;"ACRYLIC";"PUMP A";88.9;38.1;6',
    '2;"ACRYLIC";"PUMP B\nSTATION 1";60;30;5',
    '3;"PVC";"VALVE 1";50;25;4',
]


class TestGenerateLaserSheetsFFD(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = _write_csv(_E2E_CSV_ROWS, self.tmp)
        self.output_prefix = os.path.join(self.tmp, 'out', 'sheet')

    def test_returns_list_of_svg_paths(self):
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                self.csv_path, 300, 200,
                output_prefix=self.output_prefix,
                solver='ffd')
        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)

    def test_all_files_exist(self):
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                self.csv_path, 300, 200,
                output_prefix=self.output_prefix,
                solver='ffd')
        for f in files:
            self.assertTrue(os.path.exists(f), f"SVG file not found: {f}")

    def test_all_files_are_valid_svg(self):
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                self.csv_path, 300, 200,
                output_prefix=self.output_prefix,
                solver='ffd')
        for f in files:
            with open(f, encoding='utf-8') as fh:
                content = fh.read()
            self.assertIn('<svg',  content, f"{f} missing <svg>")
            self.assertIn('</svg>', content, f"{f} missing </svg>")

    def test_svg_has_all_layers(self):
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                self.csv_path, 300, 200,
                output_prefix=self.output_prefix,
                solver='ffd')
        for f in files:
            with open(f, encoding='utf-8') as fh:
                content = fh.read()
            self.assertIn('id="cut_lines"', content)
            self.assertIn('id="borders"',   content)
            self.assertIn('id="text"',      content)

    def test_two_materials_produce_separate_sheets(self):
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                self.csv_path, 300, 200,
                output_prefix=self.output_prefix,
                solver='ffd')
        # CSV has ACRYLIC and PVC → at least one sheet per material
        names = [os.path.basename(f) for f in files]
        has_acrylic = any('ACRYLIC' in n for n in names)
        has_pvc     = any('PVC' in n for n in names)
        self.assertTrue(has_acrylic, "No ACRYLIC sheet generated")
        self.assertTrue(has_pvc,     "No PVC sheet generated")

    def test_missing_csv_raises(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                generate_laser_sheets(
                    '/does/not/exist.csv', 300, 200,
                    output_prefix=self.output_prefix,
                    solver='ffd')

    def test_empty_csv_returns_empty_list(self):
        empty_csv = _write_csv([], self.tmp, 'empty.csv')
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                empty_csv, 300, 200,
                output_prefix=self.output_prefix,
                solver='ffd')
        self.assertEqual(files, [])


class TestGenerateLaserSheetsILP(unittest.TestCase):
    """ILP solver integration — short time limit so tests remain fast."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = _write_csv(_E2E_CSV_ROWS, self.tmp)
        self.output_prefix = os.path.join(self.tmp, 'out_ilp', 'sheet')

    def test_ilp_produces_valid_svgs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            files = generate_laser_sheets(
                self.csv_path, 300, 200,
                output_prefix=self.output_prefix,
                solver='ilp',
                time_limit=15)
        self.assertGreater(len(files), 0)
        for f in files:
            self.assertTrue(os.path.exists(f))
            with open(f, encoding='utf-8') as fh:
                content = fh.read()
            self.assertIn('<svg', content)

    def test_ilp_uses_fewer_or_equal_sheets_than_ffd(self):
        ffd_prefix = os.path.join(self.tmp, 'cmp_ffd', 'sheet')
        ilp_prefix = os.path.join(self.tmp, 'cmp_ilp', 'sheet')
        with contextlib.redirect_stdout(io.StringIO()):
            ffd_files = generate_laser_sheets(
                self.csv_path, 300, 200, output_prefix=ffd_prefix, solver='ffd')
            ilp_files = generate_laser_sheets(
                self.csv_path, 300, 200, output_prefix=ilp_prefix,
                solver='ilp', time_limit=15)
        self.assertLessEqual(len(ilp_files), len(ffd_files))


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
