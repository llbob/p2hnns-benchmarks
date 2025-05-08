import matplotlib as mpl

mpl.use("Agg")  # noqa
import argparse
import hashlib
import os
import math

from jinja2 import Environment, FileSystemLoader

import plot
from p2hnns_benchmarks import results
from p2hnns_benchmarks.datasets import get_dataset
from p2hnns_benchmarks.plotting.metrics import all_metrics as metrics
from p2hnns_benchmarks.plotting.plot_variants import \
    all_plot_variants as plot_variants
from p2hnns_benchmarks.plotting.utils import (compute_all_metrics,
                                           create_linestyles, create_pointset,
                                           get_plot_label)

colors = [
    "rgba(166,206,227,1)",
    "rgba(31,120,180,1)",
    "rgba(178,223,138,1)",
    "rgba(51,160,44,1)",
    "rgba(251,154,153,1)",
    "rgba(227,26,28,1)",
    "rgba(253,191,111,1)",
    "rgba(255,127,0,1)",
    "rgba(202,178,214,1)",
]

point_styles = {
    "o": "circle",
    "<": "triangle",
    "*": "star",
    "x": "cross",
    "+": "rect",
}


def convert_color(color):
    r, g, b, a = color
    return "rgba(%(r)d, %(g)d, %(b)d, %(a)d)" % {"r": r * 255, "g": g * 255, "b": b * 255, "a": a}


def convert_linestyle(ls):
    new_ls = {}
    for algo in ls.keys():
        algostyle = ls[algo]
        new_ls[algo] = (
            convert_color(algostyle[0]),
            convert_color(algostyle[1]),
            algostyle[2],
            point_styles[algostyle[3]],
        )
    return new_ls


def get_run_desc(properties):
    return "%(dataset)s_%(count)d_%(distance)s" % properties


def get_dataset_from_desc(desc):
    return desc.split("_")[0]


def get_count_from_desc(desc):
    return desc.split("_")[1]


def get_distance_from_desc(desc):
    return desc.split("_")[2]


def get_dataset_label(desc):
    return "{} (k = {})".format(get_dataset_from_desc(desc), get_count_from_desc(desc))

def get_dataset_label_wo_count(desc):
    return "{}".format(get_dataset_from_desc(desc))

def directory_path(s):
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError("'%s' is not a directory" % s)
    return s + "/"


def prepare_data(data, xn, yn):
    """Change format from (algo, instance, dict) to (algo, instance, x, y)."""
    res = []
    for algo, algo_name, result in data:
        res.append((algo, algo_name, result[xn], result[yn]))
    return res


parser = argparse.ArgumentParser()
parser.add_argument(
    "--plottype",
    help="Generate only the plots specified",
    nargs="*",
    choices=plot_variants.keys(),
    default=plot_variants.keys(),
)
parser.add_argument("--outputdir", help="Select output directory", default=".", type=directory_path, action="store")
parser.add_argument("--latex", help="generates latex code for each plot", action="store_true")
parser.add_argument("--scatter", help="create scatterplot for data", action="store_true")
parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
parser.add_argument("--groupplots", help="Generate group plots for datasets", action="store_true")
args = parser.parse_args()


def get_lines(all_data, xn, yn, render_all_points):
    """For each algorithm run on a dataset, obtain its performance
    curve coords."""
    plot_data = []
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        xs, ys, ls, axs, ays, als = create_pointset(prepare_data(all_data[algo], xn, yn), xn, yn)
        if render_all_points:
            xs, ys, ls = axs, ays, als
        plot_data.append({"name": algo, "coords": zip(xs, ys), "labels": ls, "scatter": render_all_points})
    return plot_data


def create_plot(all_data, xn, yn, linestyle, j2_env, additional_label="", plottype="line"):
    xm, ym = (metrics[xn], metrics[yn])
    render_all_points = plottype == "bubble"
    plot_data = get_lines(all_data, xn, yn, render_all_points)
    latex_code = j2_env.get_template("latex.template").render(
        plot_data=plot_data, 
        caption=get_plot_label(xm, ym), 
        xlabel=xm["description"], 
        ylabel=ym["description"],
        linestyle=linestyle,  # Pass the linestyle dictionary to the template
        point_styles=point_styles  # Pass the point_styles dictionary to the template
    )
    plot_data = get_lines(all_data, xn, yn, render_all_points)
    button_label = hashlib.sha224((get_plot_label(xm, ym) + additional_label).encode("utf-8")).hexdigest()
    return j2_env.get_template("chartjs.template").render(
        args=args,
        latex_code=latex_code,
        button_label=button_label,
        data_points=plot_data,
        xlabel=xm["description"],
        ylabel=ym["description"],
        plottype=plottype,
        plot_label=get_plot_label(xm, ym),
        label=additional_label,
        linestyle=linestyle,
        render_all_points=render_all_points,
    )

def create_groupplot_template(template_dir):
    template_path = os.path.join(template_dir, "groupplot.template")
    if os.path.exists(template_path):
        os.remove(template_path)
        
    groupplot_template = r"""\begin{figure}[htbp]
\centering
    \begin{minipage}{\textwidth}
    \centering
    \begin{tikzpicture}[scale=0.65, every mark/.append style={mark size=1.5pt}]
        % Create the group plot
        \begin{groupplot}[
            group style = {
                group size = {{ cols }} by {{ rows }},
                horizontal sep = 1.5cm,
                vertical sep = 2.5cm,
                % These settings control where labels appear
                xlabels at=edge bottom,
                ylabels at=edge left
            },
            grid = both,
            grid style = {line width=.1pt, draw=gray!30},
            major grid style = {line width=.2pt, draw=gray!50},
            height = 4.5cm,
            width = 5cm,
            xtick = {0, 0.25, 0.5, 0.75, 1},
            ymode = {{ ymode }},
            log basis y=10,
            scaled y ticks = false,
            yticklabel style={
                font=\tiny,
                /pgf/number format/.cd,
                fixed,
                precision=0
            },
            log ticks with fixed point,
            xlabel = { {{ xlabel }} },
            ylabel = { {{ ylabel }} },
            xlabel style={font=\footnotesize},
            ylabel style={font=\footnotesize},
            xticklabel style={font=\tiny},
            yticklabel style={font=\tiny},
            tick label style={font=\footnotesize},
            label style={font=\footnotesize},
            title style={yshift=0.3em, font=\footnotesize}
        ]

        
        {% for plot in group_plots %}
        \nextgroupplot[
            title = { {{ plot.title }} }
        ]
        
        {% for algo in plot.plot_data %}
        \addplot [
            color={{ algo.tikz_color }},
            mark={{ algo.tikz_mark }},
            mark size=1.5pt,
            line width=1pt
            {% if algo.scatter %},only marks{% endif %}
        ] coordinates {
            {% for coord in algo.coords %}
                ({{ coord[0]}}, {{ coord[1] }})
            {% endfor %}
        };
        {% endfor %}
        {% endfor %}
        \end{groupplot}
    \end{tikzpicture}
    \end{minipage}
    
    \begin{tikzpicture}[baseline, trim axis left, trim axis right]
        \begin{axis}[
            hide axis,
            scale only axis,
            height=0pt,
            width=0.95\textwidth,
            legend style={
                draw=none,
                fill=none,
                font=\footnotesize,
                legend columns={{ (all_algorithms|length > 8) and "5" or "4" }},
                column sep=0.8em,
                anchor=center,
                at={(0.5,0)}
            }
        ]
        {% for algo in all_algorithms %}
        \addplot[
            color={{ algo_to_color[algo] }},
            mark={{ algo_to_mark[algo] }},
            mark size=1.5pt,
            line width=1pt
        ] coordinates {(0,0)};
        \addlegendentry{ {{ algo }} };
        {% endfor %}
        \end{axis}
    \end{tikzpicture}
    
\caption{ {{ caption }} }
\label{plot:{{ caption | replace(" ", "_") | replace(",", "") | lower }}}
\end{figure}"""
    
    with open(template_path, "w") as f:
        f.write(groupplot_template)


def create_group_latex(datasets, plot_variants, linestyles, j2_env, output_dir, metrics, mode="non-batch"):
    os.makedirs(output_dir, exist_ok=True)
    
    available_datasets = list(datasets[mode].keys())
    
    if not available_datasets:
        print(f"No datasets found for {mode} mode.")
        return
    
    print(f"Found {len(available_datasets)} datasets")
    
    all_algorithms = set()
    for dataset_name in available_datasets:
        all_algorithms.update(datasets[mode][dataset_name].keys())
    all_algorithms = sorted(list(all_algorithms), key=lambda x: x.lower())
    

    tikz_colors = [

        "red!90!black", "blue!80!black", "green!70!black", "orange!90!black", 
        "violet!90!black", "teal!90!black", "magenta!90!black", "olive!90!black", 
        "cyan!70!black", "brown!90!black", "lime!70!black", "purple!80!black",

        "red!70!black", "blue!60!black", "green!50!black", "orange!70!black",
        "violet!70!black", "teal!70!black", "magenta!70!black", "olive!70!black",

        "red!30!black", "blue!40!black", "green!30!black", "orange!40!black",
        "violet!40!black", "teal!40!black", "magenta!40!black", "olive!40!black",

        "red!50!blue", "blue!50!green", "green!50!orange", "orange!50!violet",
        "violet!50!teal", "teal!50!magenta", "magenta!50!olive", "olive!50!cyan",
        "red!50!green", "blue!50!purple", "teal!50!blue", "orange!50!brown"
    ]
    

    tikz_marks = [
        "o", "square", "triangle", "diamond", "x", "pentagon", 
        "star", "otimes", "asterisk", "oplus", "+", "Mercedes star",
        "halfsquare*", "halfcircle*", "triangle*", "square*", "diamond*"
    ]
    

    algo_to_color = {algo: tikz_colors[i % len(tikz_colors)] for i, algo in enumerate(all_algorithms)}
    algo_to_mark = {algo: tikz_marks[i % len(tikz_marks)] for i, algo in enumerate(all_algorithms)}
    

    for plot_name, (xn, yn) in plot_variants.items():
        xm, ym = metrics[xn], metrics[yn]
        print(f"Creating group plot for {plot_name}: {xm['description']} vs {ym['description']}")
        
        num_plots = len(available_datasets)
        cols = min(3, num_plots)
        rows = math.ceil(num_plots / cols)

        
        group_plots = []
        for i, dataset_name in enumerate(available_datasets):

            dataset_label = get_dataset_label_wo_count(dataset_name)
            

            row = i // cols + 1
            col = i % cols + 1
            

            algo_data = []
            if dataset_name in datasets[mode]:
                for algo in all_algorithms:  # assure we use consistent order of algorithms
                    if algo in datasets[mode][dataset_name]:
                        try:
                            plot_data = prepare_data(datasets[mode][dataset_name][algo], xn, yn)
                            

                            xs, ys, ls, axs, ays, als = create_pointset(plot_data, xn, yn)
                            
                            color = algo_to_color[algo]
                            mark = algo_to_mark[algo]
                            
                            algo_data.append({
                                "name": algo,
                                "coords": list(zip(xs, ys)),
                                "tikz_color": color,
                                "tikz_mark": mark,
                                "scatter": False
                            })
                        except Exception as e:
                            print(f"Error processing algorithm {algo} for dataset {dataset_name}: {e}")
            
            group_plots.append({
                "title": dataset_label,
                "plot_data": algo_data,
                "row": row,
                "col": col
            })
        

        try:
            group_latex = j2_env.get_template("groupplot.template").render(
                group_plots=group_plots,
                rows=rows,
                cols=cols,
                xlabel=xm["description"],
                ylabel=ym["description"],
                caption=f"{plot_name.replace('/', ' vs ')} - Comparison across datasets",
                ymode="log",
                all_algorithms=all_algorithms,
                algo_to_color=algo_to_color,
                algo_to_mark=algo_to_mark
            )
            

            if "/" in plot_name:

                parts = plot_name.split("/")
                subdir = os.path.join(output_dir, parts[0])
                os.makedirs(subdir, exist_ok=True)
                
                filename = os.path.join(subdir, f"{parts[1]}_{mode}_groupplot.tex")
            else:
                filename = os.path.join(output_dir, f"{plot_name}_{mode}_groupplot.tex")
            
            with open(filename, "w") as f:
                f.write(group_latex)
            
            print(f"successfully generated group plot for {plot_name}")
        except Exception as e:
            print(f"err!! generating latex for {plot_name}: {e}")

def build_detail_site(data, label_func, j2_env, linestyles, batch=False):
    for (name, runs) in data.items():
        print("Building '%s'" % name)
        runs.keys()
        label = label_func(name)
        data = {"normal": [], "scatter": []}

        for plottype in args.plottype:
            xn, yn = plot_variants[plottype]
            data["normal"].append(create_plot(runs, xn, yn, convert_linestyle(linestyles), j2_env))
            if args.scatter:
                data["scatter"].append(
                    create_plot(runs, xn, yn, convert_linestyle(linestyles), j2_env, "Scatterplot ", "bubble")
                )

        # create png plot for summary page
        data_for_plot = {}
        for k in runs.keys():
            data_for_plot[k] = prepare_data(runs[k], "k-nn", "qps")
        plot.create_plot(
            data_for_plot, False, "linear", "log", "k-nn", "qps", args.outputdir + name + ".png", linestyles, batch
        )
        output_path = args.outputdir + name + ".html"
        with open(output_path, "w") as text_file:
            text_file.write(
                j2_env.get_template("detail_page.html").render(title=label, plot_data=data, args=args, batch=batch)
            )


def build_index_site(datasets, algorithms, j2_env, file_name):
    dataset_data = {"batch": [], "non-batch": []}
    for mode in ["batch", "non-batch"]:
        distance_measures = sorted(set([get_distance_from_desc(e) for e in datasets[mode].keys()]))
        sorted_datasets = sorted(set([get_dataset_from_desc(e) for e in datasets[mode].keys()]))

        for dm in distance_measures:
            d = {"name": dm.capitalize(), "entries": []}
            for ds in sorted_datasets:
                matching_datasets = [
                    e
                    for e in datasets[mode].keys()
                    if get_dataset_from_desc(e) == ds and get_distance_from_desc(e) == dm  # noqa
                ]
                sorted_matches = sorted(matching_datasets, key=lambda e: int(get_count_from_desc(e)))
                for idd in sorted_matches:
                    d["entries"].append({"name": idd, "desc": get_dataset_label(idd)})
            dataset_data[mode].append(d)

    with open(args.outputdir + "index.html", "w") as text_file:
        text_file.write(
            j2_env.get_template("summary.html").render(
                title="P2HNNS-Benchmarks", dataset_with_distances=dataset_data, algorithms=algorithms
            )
        )


def load_all_results():
    """Read all result files and compute all metrics"""
    all_runs_by_dataset = {"batch": {}, "non-batch": {}}
    all_runs_by_algorithm = {"batch": {}, "non-batch": {}}
    cached_true_dist = []
    old_sdn = None
    for mode in ["non-batch", "batch"]:
        for properties, f in results.load_all_results(batch_mode=(mode == "batch")):
            sdn = get_run_desc(properties)
            if sdn != old_sdn:
                dataset, _ = get_dataset(properties["dataset"])
                cached_true_dist = list(dataset["distances"])
                old_sdn = sdn
            algo_ds = get_dataset_label(sdn)
            desc_suffix = "-batch" if mode == "batch" else ""
            algo = properties["algo"] + desc_suffix
            sdn += desc_suffix
            ms = compute_all_metrics(cached_true_dist, f, properties, args.recompute)
            all_runs_by_algorithm[mode].setdefault(algo, {}).setdefault(algo_ds, []).append(ms)
            all_runs_by_dataset[mode].setdefault(sdn, {}).setdefault(algo, []).append(ms)

    return (all_runs_by_dataset, all_runs_by_algorithm)


j2_env = Environment(loader=FileSystemLoader("./templates/"), trim_blocks=True)
j2_env.globals.update(zip=zip, len=len)


if args.groupplots:
    create_groupplot_template("./templates/")

runs_by_ds, runs_by_algo = load_all_results()
dataset_names = [get_dataset_label(x) for x in list(runs_by_ds["batch"].keys()) + list(runs_by_ds["non-batch"].keys())]
algorithm_names = list(runs_by_algo["batch"].keys()) + list(runs_by_algo["non-batch"].keys())

linestyles = {**create_linestyles(dataset_names), **create_linestyles(algorithm_names)}


build_detail_site(runs_by_ds["non-batch"], lambda label: get_dataset_label(label), j2_env, linestyles, False)
build_detail_site(runs_by_ds["batch"], lambda label: get_dataset_label(label), j2_env, linestyles, True)
build_detail_site(runs_by_algo["non-batch"], lambda x: x, j2_env, linestyles, False)
build_detail_site(runs_by_algo["batch"], lambda x: x, j2_env, linestyles, True)


build_index_site(runs_by_ds, runs_by_algo, j2_env, "index.html")

# generate group plots if requested
if args.groupplots and args.latex:
    # create a 'latex' directory if it doesn't exist
    latex_dir = os.path.join(args.outputdir, "latex/")
    os.makedirs(latex_dir, exist_ok=True)
    
    for mode in ["non-batch", "batch"]:
        if runs_by_ds[mode]:  # Only if we have data for this mode
            create_group_latex(runs_by_ds, plot_variants, linestyles, j2_env, latex_dir, metrics, mode)
    
    print(f"All group plots have been generated in {latex_dir}")