"""
Interview Report Generator
============================
Generates JSON and PDF reports from session summary data.
Uses reportlab for PDF generation and matplotlib for timeline charts.
"""

import io
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "This report provides behavioral signals as supplementary interview data. "
    "It must not be used as a sole or primary hiring criterion. Human review is "
    "required. All analysis should be disclosed to candidates. Accuracy varies by "
    "lighting, camera quality, and individual differences."
)

_ETHICS_NOTE = (
    "Emotion recognition models trained on general population datasets may perform "
    "with reduced accuracy across demographic groups. Results should be interpreted "
    "with awareness of individual expression differences."
)

_DIMENSION_COLORS = {
    "cognitive_load": "#8B5CF6",
    "emotional_arousal": "#F97316",
    "engagement_level": "#14B8A6",
    "confidence_level": "#3B82F6",
}

_DIMENSION_LABELS = {
    "cognitive_load": "Cognitive Load",
    "emotional_arousal": "Emotional Arousal",
    "engagement_level": "Engagement Level",
    "confidence_level": "Confidence Level",
}

_ARC_COLORS = {
    "improving": (0.2, 0.7, 0.3),
    "declining": (0.85, 0.2, 0.2),
    "stable": (0.2, 0.4, 0.8),
    "variable": (0.9, 0.6, 0.1),
}


class InterviewReportGenerator:
    """Generates JSON and PDF interview analysis reports."""

    def generate_json(
        self,
        summary: dict,
        timeline_data: list,
        session_id: str,
        candidate_id: str = "candidate",
    ) -> dict:
        """
        Wrap a session summary into a structured JSON report envelope.
        """
        return {
            "report_version": "1.0",
            "session_id": session_id,
            "candidate_id": candidate_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": _DISCLAIMER,
            "summary": summary,
            "timeline": timeline_data,
        }

    def generate_pdf(
        self,
        summary: dict,
        timeline_data: list,
        session_id: str,
        candidate_id: str = "candidate",
        question_markers: Optional[list] = None,
    ) -> bytes:
        """
        Generate a multi-page PDF report using reportlab.

        Returns PDF as bytes.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        styles = getSampleStyleSheet()
        story = []

        # ── Page 1: Header + Summary ──
        title_style = ParagraphStyle(
            "ReportTitle", parent=styles["Heading1"], fontSize=20, spaceAfter=12
        )
        story.append(Paragraph("Behavioral Interview Analysis Report", title_style))
        story.append(Paragraph(
            f"Session: {session_id} &nbsp;|&nbsp; Candidate: {candidate_id} &nbsp;|&nbsp; "
            f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 20))

        # Dimension summary bars
        per_dim = summary.get("per_dimension", {})
        for dim_key in ["cognitive_load", "emotional_arousal", "engagement_level", "confidence_level"]:
            dim_data = per_dim.get(dim_key, {})
            mean_val = dim_data.get("mean", 0)
            label = _DIMENSION_LABELS.get(dim_key, dim_key)
            hex_color = _DIMENSION_COLORS.get(dim_key, "#888888")

            # Create a mini bar using a table
            bar_width = 300
            fill_width = max(int(mean_val * bar_width), 1)
            pct_text = f"{mean_val * 100:.1f}%"

            r, g, b = _hex_to_rgb(hex_color)
            bar_color = colors.Color(r, g, b)

            bar_data = [[label, pct_text, f"trend: {dim_data.get('trend', 0):.4f}"]]
            bar_table = Table(bar_data, colWidths=[200, 80, 150])
            bar_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), bar_color),
                ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(bar_table)
            story.append(Spacer(1, 6))

        # Overall arc badge
        arc = summary.get("overall_arc", "stable")
        arc_rgb = _ARC_COLORS.get(arc, (0.5, 0.5, 0.5))
        arc_color = colors.Color(*arc_rgb)
        story.append(Spacer(1, 12))
        arc_text = f'Overall Session Arc: <font color="#{_rgb_to_hex(*arc_rgb)}"><b>{arc.upper()}</b></font>'
        story.append(Paragraph(arc_text, styles["Heading2"]))

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Duration: {summary.get('duration_seconds', 0):.1f}s &nbsp;|&nbsp; "
            f"Frames analyzed: {summary.get('frames_analyzed', 0)} &nbsp;|&nbsp; "
            f"Baseline calibrated: {'Yes' if summary.get('baseline_calibrated') else 'No'}",
            styles["Normal"],
        ))

        # ── Page 2: Timeline + Notable Moments ──
        story.append(Spacer(1, 30))
        story.append(Paragraph("Session Timeline", styles["Heading2"]))

        # Generate matplotlib chart
        chart_bytes = self._render_timeline_chart(timeline_data, question_markers)
        if chart_bytes:
            chart_img = Image(io.BytesIO(chart_bytes), width=6.5 * inch, height=3 * inch)
            story.append(chart_img)
        story.append(Spacer(1, 12))

        # Notable moments table
        notable = summary.get("notable_moments", [])
        if notable:
            story.append(Paragraph("Notable Moments", styles["Heading3"]))
            table_data = [["Timestamp (s)", "Dimension", "Deviation"]]
            for m in notable[:20]:
                table_data.append([
                    f"{m['timestamp']:.1f}",
                    _DIMENSION_LABELS.get(m["dimension"], m["dimension"]),
                    f"{m['magnitude']:.2f}σ",
                ])
            t = Table(table_data, colWidths=[100, 180, 100])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.3)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.Color(0.95, 0.95, 0.95), colors.white]),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

        # Question breakdown table
        q_breakdowns = summary.get("question_breakdowns", {})
        if q_breakdowns:
            story.append(Paragraph("Question Breakdowns", styles["Heading3"]))
            q_header = ["Question"] + [_DIMENSION_LABELS[k] for k in ["cognitive_load", "emotional_arousal", "engagement_level", "confidence_level"]]
            q_data = [q_header]
            for qlabel, vals in q_breakdowns.items():
                row = [qlabel]
                for k in ["cognitive_load", "emotional_arousal", "engagement_level", "confidence_level"]:
                    row.append(f"{vals.get(k, 0) * 100:.1f}%")
                q_data.append(row)
            qt = Table(q_data, colWidths=[140, 90, 90, 90, 90])
            qt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.3)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(qt)

        # ── Page 3: Ethics & Disclaimer ──
        story.append(Spacer(1, 30))
        story.append(Paragraph("Ethics & Disclaimer", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(_DISCLAIMER, styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(_ETHICS_NOTE, styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"Report generated at {datetime.now(timezone.utc).isoformat()} — "
            f"Report version 1.0",
            styles["Normal"],
        ))

        doc.build(story)
        return buf.getvalue()

    def _render_timeline_chart(
        self,
        timeline_data: list,
        question_markers: Optional[list] = None,
    ) -> Optional[bytes]:
        """Render a timeline chart as PNG bytes using matplotlib."""
        if not timeline_data:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#161b22")

            times = [p["timestamp"] for p in timeline_data]

            plot_config = [
                ("cognitive_load", "#8B5CF6", "Cognitive Load"),
                ("emotional_arousal", "#F97316", "Emotional Arousal"),
                ("engagement_level", "#14B8A6", "Engagement Level"),
                ("confidence_level", "#3B82F6", "Confidence Level"),
            ]

            for key, color, label in plot_config:
                vals = [p.get(key, 0) for p in timeline_data]
                ax.plot(times, vals, color=color, linewidth=1.5, label=label, alpha=0.9)

            # Question markers
            if question_markers:
                start_ts = timeline_data[0]["timestamp"] if timeline_data else 0
                for ts, qlabel in question_markers:
                    rel_ts = ts - start_ts if start_ts else ts
                    ax.axvline(x=rel_ts, color="#ffffff", linestyle="--", alpha=0.4, linewidth=0.8)
                    ax.text(rel_ts, 1.02, qlabel, rotation=45, fontsize=6,
                            color="#aaaaaa", ha="left", va="bottom", transform=ax.get_xaxis_transform())

            ax.set_xlabel("Time (s)", color="#8b949e", fontsize=9)
            ax.set_ylabel("Score (0–1)", color="#8b949e", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(colors="#8b949e", labelsize=8)
            ax.legend(loc="upper right", fontsize=7, facecolor="#21262d",
                      edgecolor="#30363d", labelcolor="#c9d1d9")
            ax.grid(True, alpha=0.15, color="#30363d")

            for spine in ax.spines.values():
                spine.set_color("#30363d")

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue()
        except Exception as e:
            logger.warning("Failed to render timeline chart: %s", e)
            return None


def _hex_to_rgb(hex_str: str) -> tuple:
    """Convert #RRGGBB to (r, g, b) floats 0-1."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) floats 0-1 to RRGGBB hex string."""
    return f"{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
