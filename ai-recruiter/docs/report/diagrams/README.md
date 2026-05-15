# Diagram source files

Starter `.drawio` files for the PFE report. Open them at
[https://app.diagrams.net](https://app.diagrams.net) (or in VS Code with the
"Draw.io Integration" extension by `hediet`) and customise.

## Workflow

1. Open the `.drawio` file in draw.io.
2. Edit the boxes/labels/relationships.
3. Export: **File → Export as → PDF** (vector, preferred) or **PNG** (300 dpi
   for raster).
4. Save the exported asset under `docs/report/images/` with the same base
   name (e.g. `use-case.pdf`).
5. Reference it in the relevant chapter:

   ```latex
   \begin{figure}[H]
       \centering
       \includegraphics[width=0.85\textwidth]{images/use-case}
       \caption{Use-case diagram of HireFlow.}
       \label{fig:use-case}
   \end{figure}
   ```

## Files

| File | Use it in chapter | Purpose |
|---|---|---|
| `use-case.drawio` | Analysis & Design | UML use-case diagram (Recruiter + Candidate actors) |
| `class-diagram.drawio` | Analysis & Design | Domain model (Job, Candidate, Application, ScoringReport, …) |
| `sequence-jd-authoring.drawio` | Detailed Design | Sequence diagram of the JD authoring workflow with the internal crew |
| `er-diagram.drawio` | Detailed Design | Database schema (ER) |
| `deployment.drawio` | Implementation | Cloud deployment topology (browser → CDN → ALB → containers → DBs) |

## Tips

- Keep the colour palette in line with the report: blue/cyan for application
  layer, green for AI/models, orange for data, red/violet for accents.
- Use `Helvetica` or `Roboto` (not the draw.io default if it's something
  exotic) so exported PDFs read well next to LaTeX text.
- Always export as **PDF** when possible, not PNG — vectors stay sharp at any
  zoom.
- If you re-style a diagram, update all of them in one pass to keep the
  report visually consistent.
