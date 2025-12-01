"""
Tkinter GUI for the Internship Recommendation System powered by Apriori CF + ranking.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from recommender import InternshipRecommendationSystem


PLACEHOLDER_TEXT = "e.g., Python, JavaScript, React, SQL, Flask"


class InternshipRecommendationGUI:
    """Desktop interface to collect skills, train models, and show recommendations."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Internship Recommendation System")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        self.status_var = tk.StringVar(value="Ready")
        self.recommender: InternshipRecommendationSystem | None = None
        self.current_recommendations: list[dict] = []

        try:
            self.recommender = InternshipRecommendationSystem()
        except Exception as exc:  # pragma: no cover - surfaced via UI
            messagebox.showerror("Initialization Error", str(exc))

        self._build_layout()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        # Use a modern themed style
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Main.TFrame", background="#0b1020")
        style.configure("TLabel", background="#0b1020", foreground="#e4e7ef")
        style.configure("TLabelframe", background="#151b2f", foreground="#e4e7ef")
        style.configure("TLabelframe.Label", background="#151b2f", foreground="#f3b03f")
        style.configure("TButton", padding=6)
        style.map(
            "TButton",
            background=[("active", "#2d3b6e")],
            foreground=[("active", "#ffffff")],
        )

        main = ttk.Frame(self.root, padding=16, style="Main.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        # Input area
        input_frame = ttk.LabelFrame(main, text="Your Skills", padding=12)
        input_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(input_frame, text="Enter comma-separated skills:").pack(anchor=tk.W)

        self.skills_entry = ttk.Entry(input_frame, width=120)
        self.skills_entry.insert(0, PLACEHOLDER_TEXT)
        self.skills_entry.pack(fill=tk.X, pady=6)
        self.skills_entry.bind("<FocusIn>", self._clear_placeholder)
        self.skills_entry.bind("<FocusOut>", self._restore_placeholder)
        self.skills_entry.bind("<Return>", lambda _: self.request_recommendations())

        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=(8, 0))

        self.train_button = ttk.Button(button_frame, text="Train Models", command=self.request_training)
        self.train_button.pack(side=tk.LEFT, padx=4)

        self.recommend_button = ttk.Button(
            button_frame,
            text="Recommend Internships",
            command=self.request_recommendations,
        )
        self.recommend_button.pack(side=tk.LEFT, padx=4)

        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_results)
        self.clear_button.pack(side=tk.LEFT, padx=4)

        # Results area
        center_frame = ttk.Frame(main, style="Main.TFrame")
        center_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=8)

        results_frame = ttk.LabelFrame(center_frame, text="Top Matches", padding=8)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        columns = ("title", "company", "cf_score", "ranking_score", "matched", "missing", "detail")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)
        headings = {
            "title": "Internship Title",
            "company": "Company",
            "cf_score": "CF Score",
            "ranking_score": "Confidence Score",
            "matched": "Matched Skills",
            "missing": "Missing Skills",
            "detail": "Detail",
        }
        widths = {
            "title": 220,
            "company": 180,
            "cf_score": 120,
            "ranking_score": 140,
            "matched": 220,
            "missing": 220,
            "detail": 0,
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor=tk.W, stretch=col not in {"detail"})

        tree_scroll_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._handle_row_select)

        # Detail + chart panel
        side_panel = ttk.Frame(center_frame, style="Main.TFrame")
        side_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        detail_frame = ttk.LabelFrame(side_panel, text="Recommendation Details", padding=8)
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 0), pady=(0, 8))

        self.detail_text = scrolledtext.ScrolledText(
            detail_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, background="#111727", foreground="#e4e7ef"
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # Skill match chart
        chart_frame = ttk.LabelFrame(side_panel, text="Skill Match Overview", padding=8)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 0))

        self.figure = Figure(figsize=(3, 2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#111727")
        self.figure.patch.set_facecolor("#151b2f")

        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._update_chart(0, 0)

        # Status bar
        status_bar = ttk.Label(main, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, padx=4, pady=(0, 4))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _clear_placeholder(self, _event) -> None:
        if self.skills_entry.get() == PLACEHOLDER_TEXT:
            self.skills_entry.delete(0, tk.END)

    def _restore_placeholder(self, _event) -> None:
        if not self.skills_entry.get().strip():
            self.skills_entry.insert(0, PLACEHOLDER_TEXT)

    def request_training(self) -> None:
        if not self.recommender:
            messagebox.showerror("Unavailable", "Recommendation engine not initialized.")
            return

        self._set_status("Training models...")
        self._toggle_buttons(state=tk.DISABLED)

        def worker():
            try:
                summary = self.recommender.train_models(force_retrain=True)
                self.root.after(0, lambda: self._notify_success("Model training complete.", summary))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._notify_error("Training failed", e))
            finally:
                self.root.after(0, lambda: self._toggle_buttons(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def request_recommendations(self) -> None:
        skills = self.skills_entry.get().strip()
        if not skills or skills == PLACEHOLDER_TEXT:
            messagebox.showwarning("Missing Input", "Please enter your skills first.")
            return
        if not self.recommender:
            messagebox.showerror("Unavailable", "Recommendation engine not initialized.")
            return

        self._set_status("Generating recommendations...")
        self._toggle_buttons(state=tk.DISABLED)

        def worker():
            try:
                recommendations = self.recommender.recommend(skills, top_n=5)

                def update_ui():
                    self._render_recommendations(recommendations)
                    if recommendations:
                        self._set_status(f"Showing {len(recommendations)} recommendation(s).")
                    else:
                        self._set_status("No matching internships found.")

                self.root.after(0, update_ui)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._notify_error("Recommendation failed", e))
            finally:
                self.root.after(0, lambda: self._toggle_buttons(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def clear_results(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._set_detail_text("")
        self.skills_entry.delete(0, tk.END)
        self.skills_entry.insert(0, PLACEHOLDER_TEXT)
        self._update_chart(0, 0)
        self._set_status("Cleared.")

    def _handle_row_select(self, _event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        index = self.tree.index(item_id)
        if index < len(self.current_recommendations):
            rec = self.current_recommendations[index]
            self._set_detail_text(self._format_detail_text(rec))
            self._update_chart(len(rec["matched_skills"]), len(rec["missing_skills"]))

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_recommendations(self, recommendations: list[dict]) -> None:
        self.current_recommendations = recommendations or []
        self.tree.delete(*self.tree.get_children())
        if not recommendations:
            self._set_detail_text("No recommendations to display.")
            self._update_chart(0, 0)
            return

        for rec in recommendations:
            detail_text = self._format_detail_text(rec)
            self.tree.insert(
                "",
                tk.END,
                values=(
                    rec["internship_title"],
                    rec["company"],
                    rec["cf_score"],
                    rec["ranking_score"],
                    ", ".join(rec["matched_skills"]) or "-",
                    ", ".join(rec["missing_skills"]) or "-",
                    detail_text,
                ),
            )
        first = self.tree.get_children()
        if first:
            self.tree.selection_set(first[0])
            self._handle_row_select(None)

    def _format_detail_text(self, rec: dict) -> str:
        lines = [
            f"Internship: {rec['internship_title']} ({rec['company']})",
            f"Location: {rec.get('location', 'N/A')} | Experience: {rec.get('minimum_experience', '0')} year(s)",
            f"Final Score: {rec['final_score']} | Match Score: {rec['match_percentage']}%",
            f"CF Score: {rec['cf_score']} | Confidence Score: {rec['ranking_score']}",
            "",
            "Required Skills:",
            ", ".join(rec["required_skills"]) or "-",
            "",
            "Preferred Skills:",
            ", ".join(rec["preferred_skills"]) or "-",
            "",
            "Matched Skills:",
            ", ".join(rec["matched_skills"]) or "-",
            "",
            "Missing Skills:",
            ", ".join(rec["missing_skills"]) or "-",
        ]
        return "\n".join(lines)

    def _set_detail_text(self, text: str) -> None:
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text)
        self.detail_text.configure(state=tk.DISABLED)

    def _update_chart(self, matched_count: int, missing_count: int) -> None:
        """Render a simple donut chart of matched vs missing skills."""
        self.ax.clear()
        self.ax.set_facecolor("#111727")
        labels = ["Matched", "Missing"]
        values = [matched_count, missing_count]
        colors = ["#3cb371", "#e55353"]

        if matched_count == 0 and missing_count == 0:
            self.ax.text(
                0.5,
                0.5,
                "No skills to display",
                ha="center",
                va="center",
                color="#e4e7ef",
                fontsize=9,
            )
        else:
            wedges, _ = self.ax.pie(
                values,
                labels=labels,
                colors=colors,
                startangle=140,
                wedgeprops={"width": 0.5},
                textprops={"color": "#e4e7ef", "fontsize": 8},
            )
            self.ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=8)

        self.ax.set_aspect("equal")
        self.chart_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _toggle_buttons(self, state: str) -> None:
        for button in (self.train_button, self.recommend_button, self.clear_button):
            button.configure(state=state)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _notify_success(self, message: str, summary: dict) -> None:
        self._set_status(message)
        detail = "\n".join(f"{key}: {value}" for key, value in summary.items())
        messagebox.showinfo("Success", f"{message}\n\n{detail}")

    def _notify_error(self, title: str, exc: Exception) -> None:
        self._set_status(f"{title}. See details.")
        messagebox.showerror(title, str(exc))


