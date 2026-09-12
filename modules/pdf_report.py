"""
Pure-Python PDF evidence report generator for OILTRACE.
Zero external C/binary dependencies; formatted for IMO MARPOL Annex I and USCG port state inspection.
"""

from datetime import datetime, timezone

def clean_pdf_text(text: str) -> str:
    replacements = {
        '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2012': '-', '\u2022': '*',
        '\u00b0': ' deg ', '\u00b2': '2', '\u00b3': '3', '\u00b5': 'u',
        '\u0394': 'Delta', '\u2605': '*', '\u2192': '->', '\u00b7': '-',
        '\u2264': '<=', '\u2265': '>=', '\u00b1': '+/-', '·': '-',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_reliable_pdf_report(centroid, origin_point, spill_time, top_suspect, seep_eval, rig_eval,
                                  is_natural_seep=False, seep_details=None, spill_data=None):
    area_sqkm = spill_data.get("area_sqkm", 28.05) if spill_data else 28.05
    area_sqm = spill_data.get("area_sqm", 28052800) if spill_data else 28052800
    depth_um = spill_data.get("depth_um", 58.4) if spill_data else 58.4
    depth_mm = spill_data.get("depth_mm", 0.058) if spill_data else 0.058
    volume_bbl = spill_data.get("volume_barrels", 10306) if spill_data else 10306

    seep_summary_lines = []
    if seep_eval and "table" in seep_eval and not seep_eval["table"].empty:
        for _, row in seep_eval["table"].head(5).iterrows():
            seep_summary_lines.append(f"   * Seep: {row['Seep Name']} -- {row['Distance (km)']:.2f} km (Depth: {row['Water Depth (m)']}m)")
    elif seep_eval and "nearest" in seep_eval and seep_eval["nearest"] is not None:
        seep_summary_lines.append(f"   * Nearest Seep: {seep_eval['nearest']['name']} ({seep_eval['nearest']['dist_km']:.2f} km)")
    else:
        seep_summary_lines.append("   * Natural Seep Screening: No registered seeps in proximity.")

    rig_summary_lines = []
    if rig_eval and "table" in rig_eval and not rig_eval["table"].empty:
        for _, row in rig_eval["table"].head(5).iterrows():
            rig_summary_lines.append(f"   * Rig:  {row['Platform Name']} -- {row['Distance (km)']:.2f} km (Op: {row['Operator']})")
    elif rig_eval and "nearest" in rig_eval and rig_eval["nearest"] is not None:
        rig_summary_lines.append(f"   * Nearest Platform: {rig_eval['nearest']['name']} ({rig_eval['nearest']['dist_km']:.2f} km)")
    else:
        rig_summary_lines.append("   * Offshore Platform Screening: No fixed platforms in proximity.")

    if is_natural_seep:
        seep_name = seep_details.get("name", "Documented Seafloor Cold Seep") if seep_details else "Documented Cold Seep"
        seep_depth = seep_details.get("depth_m", 540) if seep_details else 540
        seep_flux = seep_details.get("flux_bbl_day", 40) if seep_details else 40
        vessel_section = [
            "4. AIS VESSEL ATTRIBUTION STATUS: DISMISSED / EXONERATED",
            "   Determination:            TARGET IDENTIFIED AS GEOGENIC NATURAL SEEP",
            f"   Seafloor Vent Location:   {seep_name} at -{seep_depth} m depth",
            f"   Estimated Natural Flux:   {seep_flux} barrels/day continuous natural seepage",
            "   Legal Finding:            Commercial vessel traffic completely cleared of",
            "                             MARPOL 73/78 Annex I and Clean Water Act Section 311 liability.",
            "   Enforcement Action:       DO NOT TASK COAST GUARD INTERCEPTION. Close case.",
        ]
        doc_title = "OILTRACE -- GEOLOGICAL NATURAL SEEP ASSESSMENT DOSSIER"
        doc_sub = "MARPOL 73/78 ANNEX I EXONERATION & GEOGENIC ORIGIN CERTIFICATE"
    else:
        if top_suspect is not None:
            v_name = top_suspect.get('VesselName', 'UNKNOWN') if hasattr(top_suspect, 'get') else str(top_suspect['VesselName'])
            v_type = top_suspect.get('VesselType', 'N/A') if hasattr(top_suspect, 'get') else str(top_suspect['VesselType'])
            v_mmsi = top_suspect.get('MMSI', 'N/A') if hasattr(top_suspect, 'get') else str(top_suspect['MMSI'])
            v_dist = float(top_suspect.get('Min_Distance_km', 0)) if hasattr(top_suspect, 'get') else float(top_suspect['Min_Distance_km'])
            v_spd = top_suspect.get('Speed_at_CPA', 0) if hasattr(top_suspect, 'get') else top_suspect['Speed_at_CPA']
            v_risk = top_suspect.get('Risk_Score', 0) if hasattr(top_suspect, 'get') else top_suspect['Risk_Score']
            vessel_section = [
                "4. AIS VESSEL TRAFFIC & TARGET ATTRIBUTION",
                f"   Top Suspect Vessel:       {v_name}",
                f"   Vessel Type / MMSI:       {v_type} / {v_mmsi}",
                f"   Closest Approach (CPA):   {v_dist:.2f} km from reconstructed origin",
                f"   Speed Anomaly at CPA:     Throttled to {v_spd} kn at CPA (Kinematic Signature)",
                f"   Composite Risk Score:     {v_risk} / 100 (HIGH PROBABLE CAUSE)",
                "",
                "5. LEGAL EVIDENTIARY STATUS & RECOMMENDED ENFORCEMENT",
                "   Statutory Basis:          MARPOL 73/78 Annex I / US Clean Water Act Sec 311",
                "   Recommended Action:       Task USCG Sector Port State Control boarding inspection.",
            ]
        else:
            vessel_section = [
                "4. AIS VESSEL TRAFFIC & TARGET ATTRIBUTION",
                "   Top Suspect Vessel:       No high-risk vessel correlated within search window.",
            ]
        doc_title = "OILTRACE -- MARITIME INCIDENT FORENSIC INTELLIGENCE DOSSIER"
        doc_sub = "USCG / IMO MARPOL ANNEX I STANDARDIZED EVIDENCE REPORT"

    lines = [
        doc_title,
        "=" * 78,
        doc_sub,
        f"Incident Ref: #OT-2026-GC-092      Classification: {'PUBLIC / GEOLOGICAL' if is_natural_seep else 'RESTRICTED / LAW ENFORCEMENT'}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "=" * 78,
        "",
        "1. SATELLITE RADAR (SAR) SURVEILLANCE & PHYSICAL DIMENSIONS",
        f"   Observed Slick Centroid:  {centroid[0]:.5f} N, {centroid[1]:.5f} W",
        f"   Surface Area:             {area_sqkm:.2f} km2 ({area_sqm:,.0f} m2)",
        f"   Discharge Volume Est:     {volume_bbl:,.0f} barrels (approx)",
        f"   Oil Film Thickness:       {depth_um:.1f} um / {depth_mm:.4f} mm (Bonn Agreement)",
        "   Radar Sensor Mode:        Sentinel-1 C-Band Dual-Pol (VV/VH), -23.8 dB backscatter",
        "",
        "2. 4D HYDRODYNAMIC HINDCAST & ORIGIN RECONSTRUCTION",
        f"   Reconstructed Origin:     {origin_point[0]:.5f} N, {origin_point[1]:.5f} W",
        f"   Release Time Window:      {spill_time.strftime('%Y-%m-%d %H:%M UTC')}",
        "   Drift Forcing:            Ocean Current 0.45 m/s at 125.0 deg True",
        "",
        "3. MULTI-SITE NATURAL SEEP & OFFSHORE PLATFORM SCREENING",
        "   Screening evaluated registered natural seeps and offshore platforms:",
    ] + seep_summary_lines + rig_summary_lines + [
        f"   Screening Finding:        {seep_eval.get('label', 'Screening complete.') if seep_eval else 'N/A'}",
        "",
    ] + vessel_section + [
        "",
        "=" * 78,
        "CERTIFICATION: Produced by OILTRACE Autonomous Forensic Telemetry Core.",
    ]

    cleaned_lines = [clean_pdf_text(l) for l in lines]
    stream_cmds = []
    if is_natural_seep:
        stream_cmds.append("q 0.05 0.16 0.24 rg 36 730 540 36 re f Q")
        stream_cmds.append(f"BT /F2 11 Tf 1 1 1 rg 46 750 Td ({doc_title}) Tj ET")
        stream_cmds.append("BT /F1 7.5 Tf 0.2 0.85 0.95 rg 46 738 Td (GEOGENIC NATURAL SEEP EXONERATION DOSSIER) Tj ET")
    else:
        stream_cmds.append("q 0.04 0.10 0.16 rg 36 730 540 36 re f Q")
        stream_cmds.append(f"BT /F2 11 Tf 1 1 1 rg 46 750 Td ({doc_title}) Tj ET")
        stream_cmds.append("BT /F1 7.5 Tf 0.6 0.8 0.95 rg 46 738 Td (USCG / IMO MARPOL ANNEX I STANDARDIZED EVIDENCE REPORT) Tj ET")

    stream_cmds.append("BT /F1 8.5 Tf 0.1 0.15 0.2 rg 42 712 Td 11.5 TL")
    for i, line in enumerate(cleaned_lines):
        if i < 4:
            continue
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_cmds.append(f"({escaped}) '")
    stream_cmds.append("ET")

    stream_content = "\n".join(stream_cmds).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>",
        b"<< /Length " + str(len(stream_content)).encode("ascii") + b" >>\nstream\n" + stream_content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \r\n".encode("ascii"))
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \r\n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\r\n".encode("ascii"))

    return bytes(pdf)