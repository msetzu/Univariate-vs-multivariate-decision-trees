from bokeh.models import Range1d, Band, ColumnDataSource, Label
from bokeh.plotting import figure
from bokeh.models import Legend, LegendItem


import pandas
import numpy


def draw_twovariate_dataset(data, flipped=None, width=900, height=450):
    plt = figure(
        x_axis_label = "X0",
        y_axis_label = "X1",
        frame_width=width, frame_height=height,
        toolbar_location=None
    )

    labels = data[:, -1]

    if flipped is None:
        x0_class_0, x0_class_1 = data[labels == 0, 0], data[labels == 1, 0]
        x1_class_0, x1_class_1 = data[labels == 0, 1], data[labels == 1, 1]
        
        plt.scatter(x0_class_0, x1_class_0, color="red", size=10, marker="circle", fill_alpha=0.1)
        plt.scatter(x0_class_1, x1_class_1, color="blue", size=10, marker="triangle", fill_alpha=0.1)
    else:
        flipped_data = data[flipped]
        non_flipped_data = data[~flipped]
        flipped_labels = labels[flipped]
        non_flipped_labels = labels[~flipped]

        x0_class_0, x0_class_1 = non_flipped_data[non_flipped_labels == 0, 0], non_flipped_data[non_flipped_labels == 1, 0]
        x1_class_0, x1_class_1 = non_flipped_data[non_flipped_labels == 0, 1], non_flipped_data[non_flipped_labels == 1, 1]
        flipped_x0_class_0, flipped_x0_class_1 = flipped_data[flipped_labels == 0, 0], flipped_data[flipped_labels == 1, 0]
        flipped_x1_class_0, flipped_x1_class_1 = flipped_data[flipped_labels == 0, 1], flipped_data[flipped_labels == 1, 1]
        
        plt.scatter(x0_class_0, x1_class_0, color="red", size=10, marker="circle", fill_alpha=0.1)
        plt.scatter(x0_class_1, x1_class_1, color="blue", size=10, marker="triangle", fill_alpha=0.1)
        # flipped data
        plt.scatter(flipped_x0_class_0, flipped_x1_class_0, color="green", size=10, marker="triangle", fill_alpha=0.5)
        plt.scatter(flipped_x0_class_1, flipped_x1_class_1, color="green", size=10, marker="triangle", fill_alpha=0.5)
    
    plt.xaxis.major_label_text_font_size = "50pt"
    plt.xaxis.axis_label_text_font_size = "50pt"
    plt.xaxis.axis_label_text_font_size = "50pt"
    plt.yaxis.major_label_text_font_size = "50pt"
    plt.yaxis.axis_label_text_font_size = "50pt"

    plt.x_range = Range1d(-4, +4)
    plt.y_range = Range1d(-4, +4)
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    
    return plt


def draw_slope_frequency(slopes, frequencies, width=900, height=450):
    plt = figure(
        x_axis_label = "slope",
        y_axis_label = "frequency",
        frame_width=width, frame_height=height,
    )

    plt.scatter(slopes, frequencies, marker="triangle", size=10, fill_alpha=0.5, color="red")
    plt.xaxis.major_label_text_font_size = "50pt"
    plt.xaxis.axis_label_text_font_size = "50pt"
    plt.xaxis.axis_label_text_font_size = "50pt"
    plt.yaxis.major_label_text_font_size = "50pt"
    plt.yaxis.axis_label_text_font_size = "50pt"
    
    return plt


def performance_by_correlation(correlation, y_univariate, y_multivariate, y_univariate_std, y_multivariate_std,
                               performance_metric, noise, width=900, height=450):
    colors = ["red", "blue"]

    plt = figure(#title=f"{performance_metric.capitalize()} on increasing correlation (label noise = {noise})",
                 x_axis_label=f"ρ on ε = {noise}",
                 y_axis_label=f"{performance_metric}",
                 frame_width=width, frame_height=height,
                 toolbar_location=None)

    # for uni, multi, uni_std, multi_std, name, c in zip(y_univariate, y_multivariate, y_univariate_std, y_multivariate_std, metrics, colors):
    plt.line(correlation, y_univariate, line_width=10, color=colors[0], line_dash="dotted")
    plt.line(correlation, y_multivariate, line_width=10, color=colors[1], line_dash="dotted")
    plt.scatter(correlation, y_univariate, color=colors[0], fill_alpha=1, size=100, marker="triangle", fill_color=colors[0])
    plt.scatter(correlation, y_multivariate, color=colors[1], fill_alpha=1, size=100, marker="square", fill_color=colors[1])
    
    df = pandas.DataFrame(data={
                                "corr": correlation,

                                "uni": y_univariate, "std_uni": y_univariate_std, 
                                "lower_univariate": y_univariate - y_univariate_std,
                                "upper_univariate": y_univariate + y_univariate_std,
                                
                                "multi": y_multivariate, "std_multi": y_multivariate_std,
                                "lower_multivariate": y_multivariate - y_multivariate_std,
                                "upper_multivariate": y_multivariate + y_multivariate_std,
                               })

    source = ColumnDataSource(df)
    band_univariate = Band(base="corr", lower="lower_univariate", upper="upper_univariate", source=source,
                            fill_alpha=0.1, fill_color=colors[0], line_color=colors[0])
    band_multivariate = Band(base="corr", lower="lower_multivariate", upper="upper_multivariate", source=source,
                             fill_alpha=0.1, fill_color=colors[1], line_color=colors[1])

    plt.add_layout(band_univariate)
    plt.add_layout(band_multivariate)

    min_y = min([df.lower_univariate.min(), df.lower_multivariate.min()])
    max_y = max([df.upper_univariate.max(), df.upper_multivariate.max()])
    plt.x_range = Range1d(.1, 1.05)
    # plt.y_range = Range1d(min_y - min_y * 0.1, max_y + max_y * 0.09)
    plt.y_range = Range1d(0.5, 1.05)
    # plt.legend.orientation = "horizontal"
    # plt.legend.location = "bottom_left"
    # plt.legend.click_policy = "hide"
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.xaxis.major_label_text_font_size = "100pt"
    plt.xaxis.axis_label_text_font_size = "100pt"
    plt.xaxis.axis_label_text_font_size = "100pt"
    
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "100pt"
    plt.yaxis.axis_label_text_font_size = "100pt"

    if noise != 0.:
        plt.xaxis.major_label_text_font_size = "100pt"
        plt.xaxis.axis_label_text_font_size = "100pt"
        plt.xaxis.axis_label_text_font_size = "100pt"
        plt.yaxis.major_label_text_font_size = "0pt"
        plt.yaxis.axis_label_text_font_size = "0pt"

    if performance_metric != "Acc":
        plt.xaxis.major_label_text_font_size = "0pt"
        plt.xaxis.axis_label_text_font_size = "0pt"
        plt.xaxis.axis_label_text_font_size = "0pt"

    plt.title.text_font_size = "0pt"
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].axis_line_width = 5
    plt.yaxis[0].axis_line_width = 5
    
    return plt


def draw_mean_correlations(mean_correlations, std_correlations, width=900, height=450):   
    plt = figure(title=f"Mean correlation per dataset.",
                 x_axis_label="Dataset", y_axis_label=f"Correlation",
                 width=width, height=height)
    plt.x_range = Range1d(0, len(mean_correlations))
    # plt.y_range = Range1d(-1, max(mean_correlations + std_correlations) + 0.1)

    # for uni, multi, uni_std, multi_std, name, c in zip(y_univariate, y_multivariate, y_univariate_std, y_multivariate_std, metrics, colors):
    sorting_indexes = numpy.argsort(mean_correlations)
    y = mean_correlations[sorting_indexes]
    x = numpy.arange(len(mean_correlations))

    plt.line(x, y, legend_label="Correlation", line_width=3, color="red")
    
    df = pandas.DataFrame(data={
                                "x": x, "y": y,
                                "lower": y - std_correlations[sorting_indexes],
                                "upper": y + std_correlations[sorting_indexes],
                               })

    source = ColumnDataSource(df)
    band_std = Band(base="x", lower="lower", upper="upper", source=source,
                    fill_alpha=0.1, fill_color="red", line_color="red")
    plt.add_layout(band_std)

    plt.legend.orientation = "horizontal"
    plt.legend.location = "bottom_left"
    plt.legend.click_policy = "hide"
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "30pt"
    plt.xaxis.major_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    plt.yaxis.axis_label_text_font_size = "30pt"
    plt.title.text_font_size = "56pt"
    
    return plt


def size_difference_in_multiplier_per_noise(correlations, y_univariates, y_multivariates, width=900, height=450):
    colors = ["blue",
              "purple", "green", "yellow", "orange", "red"]
    markers = ["triangle", "square", "star", "hex", "circle"]
    
    plt = figure(title=f"Node count multiplier on increasing correlation",
                 x_axis_label="Correlation", y_axis_label=f"Size ratio",
                 frame_width=width, frame_height=height,
                 toolbar_location=None
                 )

    min_y, max_y = numpy.inf, -numpy.inf
    for i, (y_univariate, y_multivariate, c) in enumerate(zip(y_univariates, y_multivariates, colors)):
        y = y_univariate / y_multivariate
        plt.line(correlations, y, line_width=5, color=c, line_dash="dashed")

        min_y = min_y if y.min() > min_y else y.min()
        max_y = max_y if y.max() < max_y else y.max()

    legend_it = []
    for i, (y_univariate, y_multivariate, c, marker) in enumerate(zip(y_univariates, y_multivariates, colors, markers)):
        y = y_univariate / y_multivariate
        l = plt.scatter(correlations, y, line_width=10, color=c, marker=marker, fill_color="white", fill_alpha=.5, size=100)
        legend_it.append(LegendItem(label="ε = 0." + str(i), renderers=[l]))

    legend = Legend(items=legend_it)
    legend.click_policy="mute"
    legend.orientation = "horizontal"
    legend.glyph_height = 200
    legend.glyph_width = 200
    legend.label_text_font_size = "120pt"
    plt.add_layout(legend, "above")
   
    plt.y_range = Range1d(0, max_y * 11.1)
    # plt.x_range = Range1d(0.1, 0.905)
    plt.x_range = Range1d(0.1, 10)
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.xaxis.major_label_text_font_size = "80pt"
    plt.xaxis.axis_label_text_font_size = "80pt"
    plt.xaxis.axis_label_text_font_size = "0pt"
    
    plt.yaxis.major_label_text_font_size = "80pt"
    plt.yaxis.axis_label_text_font_size = "80pt"
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    
    plt.title.text_font_size = "0pt"
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].axis_line_width = 5
    plt.yaxis[0].axis_line_width = 5
    
    return plt


def size_difference_absolute_per_noise(y_univariates, y_multivariates, width=900, height=450):
    correlation = [i / 10 for i in range(1, 10)]
    colors = ["blue", "purple", "green", "yellow", "orange", "red"]
    
    plt = figure(title=f"Absolute node count difference on increasing correlation",
                 x_axis_label="Correlation", y_axis_label=f"Size multiplicator",
                 width=width, height=height, toolbar_location=None)
    plt.x_range = Range1d(0.1, 0.9)

    min_y, max_y = numpy.inf, -numpy.inf
    for i, (y_univariate, y_multivariate, c) in enumerate(zip(y_univariates, y_multivariates, colors)):
        y = y_univariate - y_multivariate
        plt.line(correlation, y, line_width=3, color=c, legend_label=str(i / 10))

        min_y = min_y if y.min() > min_y else y.min()
        max_y = max_y if y.max() < max_y else y.max()
   
    plt.y_range = Range1d(min_y, max_y)
    plt.x_range = Range1d(0.1, 0.9)
    plt.legend.orientation = "vertical"
    plt.legend.location = "bottom_left"
    plt.legend.click_policy = "hide"
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "40pt"
    plt.xaxis.major_label_text_font_size = "40pt"
    plt.xaxis.axis_label_text_font_size = "40pt"
    plt.xaxis.axis_label_text_font_size = "0pt"
    plt.yaxis.axis_label_text_font_size = "0pt"
    plt.title.text_font_size = "30pt"
    plt.yaxis[0].ticker.desired_num_ticks = 3
    plt.xaxis[0].ticker.desired_num_ticks = 4

       
    return plt


def sizes_per_noise(correlations, y_univariates, y_multivariates, width=900, height=450):
    colors = ["blue", "purple", "green", "yellow", "orange", "red"]
    markers = ["triangle", "square", "star", "hex", "circle"]
    
    plt = figure(title=f"Absolute node count difference on increasing correlation",
                 x_axis_label="Correlation", y_axis_label=f"Tree size",
                 frame_width=width, frame_height=height, toolbar_location=None)
    plt.x_range = Range1d(0.1, 0.9)

    min_y, max_y = numpy.inf, -numpy.inf
    for i, (y_univariate, y_multivariate, c) in enumerate(zip(y_univariates, y_multivariates, colors)):
        # plt.line(correlation, y_univariate, line_width=3, color=c, legend_label=str(i / 10))
        # plt.line(correlation, y_multivariate, line_width=3, color=c, legend_label=str(i / 10), line_dash="dashed")
        plt.line(correlations, y_univariate, line_width=5, color=c, line_dash="dotted")
        plt.line(correlations, y_multivariate, line_width=5, color=c)
        y = numpy.hstack((y_univariate, y_multivariate))

        min_y = min_y if y.min() > min_y else y.min()
        max_y = max_y if y.max() < max_y else y.max()

    for i, (y_univariate, y_multivariate, c, marker) in enumerate(zip(y_univariates, y_multivariates, colors, markers)):
        # plt.line(correlation, y_univariate, line_width=3, color=c, legend_label=str(i / 10))
        # plt.line(correlation, y_multivariate, line_width=3, color=c, legend_label=str(i / 10), line_dash="dashed")
        plt.scatter(correlations, y_univariate, line_width=10, color=c, marker=marker, fill_alpha=1., size=100, fill_color="white")
        plt.scatter(correlations, y_multivariate, line_width=10, color=c, marker=marker, fill_alpha=0.5, size=100, fill_color=c)
        y = numpy.hstack((y_univariate, y_multivariate))
   
    plt.y_range = Range1d(-10, max_y + max_y * 0.1)
    # plt.legend.orientation = "horizontal"
    # plt.legend.location = "top_left"
    # plt.legend.click_policy = "hide"
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    
    plt.title.text_font_size = "0pt"
    plt.xaxis.major_label_text_font_size = "80pt"
    plt.xaxis.axis_label_text_font_size = "80pt"
    plt.xaxis.axis_label_text_font_size = "0pt"
    
    plt.yaxis.major_label_text_font_size = "80pt"
    plt.yaxis.axis_label_text_font_size = "80pt"
    plt.yaxis[0].ticker.desired_num_ticks = 6
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].axis_line_width = 5
    plt.yaxis[0].axis_line_width = 4
    
    # plt.legend.title_text_font_size = "80pt"
    # plt.legend.label_text_font_size = "80pt"
    plt.yaxis.axis_label_text_font_size = "80pt"

    return plt


def sizes_per_noise_by_slope(correlations, y_univariates, y_multivariates, markers, colors, noises, width=900, height=450):    
    plt = figure(title=f"",
                 x_axis_label="Correlation",
                 y_axis_label=f"Tree size",
                 frame_width=width, frame_height=height, toolbar_location=None)
    plt.x_range = Range1d(0.1, 1.)

    min_y, max_y = numpy.inf, -numpy.inf
    legend_uni = list()
    legend_multi = list()
    for i, (eps, y_univariate, y_multivariate, c) in enumerate(zip(noises, y_univariates, y_multivariates, colors)):
        # plt.line(correlation, y_univariate, line_width=3, color=c, legend_label=str(i / 10))
        # plt.line(correlation, y_multivariate, line_width=3, color=c, legend_label=str(i / 10), line_dash="dashed")
        plt.line(correlations, y_univariate, line_width=5, line_alpha=0.5, color=colors[i][0])
        plt.line(correlations, y_multivariate, line_width=5, line_alpha=0.5, color=colors[i][1])
        l1 = plt.scatter(correlations, y_univariate, size=30, fill_alpha=.75, marker=markers[i][0], color=colors[i][0])
        l2 = plt.scatter(correlations, y_multivariate, size=30, fill_alpha=.75, marker=markers[i][1], color=colors[i][1])
        
        legend_uni.append(LegendItem(label=f"ε={eps}", renderers=[l1]))
        legend_multi.append(LegendItem(label=f"ε={eps}", renderers=[l2]))
        # y = numpy.hstack((y_univariate, y_multivariate))

        # min_y = min_y if y.min() > min_y else y.min()
        # max_y = max_y if y.max() < max_y else y.max()
  
    # plt.y_range = Range1d(-10, max_y + max_y * 0.1)
    # plt.legend.orientation = "horizontal"
    # plt.legend.location = "top_left"
    # plt.legend.click_policy = "hide"

    legend = Legend(items=legend_uni + legend_multi)
    legend.click_policy="mute"
    legend.orientation = "horizontal"
    legend.glyph_height = 200
    legend.glyph_width = 200
    legend.label_text_font_size = "40pt"
    # plt.add_layout(legend, "above")

    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    
    plt.title.text_font_size = "0pt"
    plt.xaxis.major_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    
    plt.yaxis.major_label_text_font_size = "30pt"
    plt.yaxis.axis_label_text_font_size = "30pt"
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    
    # plt.legend.title_text_font_size = "80pt"
    # # plt.legend.label_text_font_size = "80pt"
    # plt.yaxis.axis_label_text_font_size = "80pt"

    return plt


def sizes_per_noise_by_slope_on_row(correlations, full_y_univariates, full_y_multivariates, markers, colors, noises, width=900, height=450):    
    plts = list()
    thetas = [0, 15, 30, 45]
    for i_slope, (theta, y_univariates, y_multivariates) in enumerate(zip(thetas, full_y_univariates, full_y_multivariates)):
        plt = figure(title=f"",
                    x_axis_label=f"ρ on θ = {theta}°",
                    y_axis_label=f"Tree size",
                    frame_width=width, frame_height=height, toolbar_location=None)
        plt.x_range = Range1d(0.1, 1.)

        min_y, max_y = numpy.inf, -numpy.inf
        legend_uni = list()
        legend_multi = list()
        for i, (eps, y_univariate, y_multivariate, c) in enumerate(zip(noises, y_univariates, y_multivariates, colors)):
            # plt.line(correlation, y_univariate, line_width=3, color=c, legend_label=str(i / 10))
            # plt.line(correlation, y_multivariate, line_width=3, color=c, legend_label=str(i / 10), line_dash="dashed")
            plt.line(correlations, y_univariate, line_width=5, line_alpha=0.5, color=colors[i][0])
            plt.line(correlations, y_multivariate, line_width=5, line_alpha=0.5, color=colors[i][1])
            l1 = plt.scatter(correlations, y_univariate, size=30, fill_alpha=.75, marker=markers[i][0], color=colors[i][0])
            l2 = plt.scatter(correlations, y_multivariate, size=30, fill_alpha=.75, marker=markers[i][1], color=colors[i][1])
            
            legend_uni.append(LegendItem(label=f"ε={eps}", renderers=[l1]))
            legend_multi.append(LegendItem(label=f"ε={eps}", renderers=[l2]))
            # y = numpy.hstack((y_univariate, y_multivariate))

            # min_y = min_y if y.min() > min_y else y.min()
            # max_y = max_y if y.max() < max_y else y.max()
    
        # plt.y_range = Range1d(-10, max_y + max_y * 0.1)
        # plt.legend.orientation = "horizontal"
        # plt.legend.location = "top_left"
        # plt.legend.click_policy = "hide"

        legend = Legend(items=legend_uni + legend_multi)
        legend.click_policy="mute"
        legend.orientation = "horizontal"
        legend.glyph_height = 200
        legend.glyph_width = 200
        legend.label_text_font_size = "40pt"
        # plt.add_layout(legend, "above")

        plt.xaxis.axis_label_text_font = "CMU"
        plt.yaxis.axis_label_text_font = "CMU"
        plt.title.text_font = "CMU"
        plt.xgrid.grid_line_color = None
        plt.ygrid.grid_line_color = None
        plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        plt.yaxis[0].ticker.desired_num_ticks = 5
        plt.xaxis[0].ticker.desired_num_ticks = 5
        plt.yaxis[0].ticker.desired_num_ticks = 5
        plt.xaxis[0].ticker.desired_num_ticks = 5
        

        plt.title.text_font_size = "0pt"
        plt.xaxis.major_label_text_font_size = "30pt"
        plt.xaxis.axis_label_text_font_size = "30pt"
        plt.xaxis.axis_label_text_font_size = "30pt"
        if i_slope == 0:
            plt.yaxis.major_label_text_font_size = "30pt"
            plt.yaxis.axis_label_text_font_size = "30pt"
        else:
            plt.yaxis.major_label_text_font_size = "0pt"
            plt.yaxis.axis_label_text_font_size = "0pt"
        
        plts.append(plt)
    
    # plt.legend.title_text_font_size = "80pt"
    # # plt.legend.label_text_font_size = "80pt"
    # plt.yaxis.axis_label_text_font_size = "80pt"

    return plts

def draw_correlations(pearsons, taus, spearmans, width=900, height=450):   
    pearsons_vals, pearsons_counts = numpy.unique(pearsons[~numpy.isnan(pearsons)], return_counts=True)
    taus_vals, taus_counts = numpy.unique(pearsons[~numpy.isnan(pearsons)], return_counts=True)
    spearman_vals, spearman_counts = numpy.unique(spearmans[~numpy.isnan(spearmans)], return_counts=True)

    plt = figure(title=f"Correlation distribution on benchmark datasets",
                 x_axis_label="Correlation", y_axis_label=f"Frequency",
                 width=width, height=height)
    plt.x_range = Range1d(-1., +1.)

    plt.line(pearsons_vals, pearsons_counts, line_width=3, color="red", legend_label="Pearson")
    plt.line(taus_vals, taus_counts, line_width=3, color="blue", legend_label="Tau")
    plt.line(spearman_vals, spearman_counts, line_width=3, color="green", legend_label="Spearman")
    
    plt.legend.orientation = "vertical"
    plt.legend.location = "top_left"
    plt.legend.click_policy = "hide"
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "30pt"
    plt.xaxis.major_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    plt.yaxis.axis_label_text_font_size = "30pt"
    plt.title.text_font_size = "56pt"
    
    return plt


def draw_correlations_cdf(correlations, width=900, height=450, label=""):   
    plt = figure(title=f"",
                 x_axis_label=label, y_axis_label=f"CDF",
                 width=width, height=height,
                 toolbar_location=None)
    # plt.toolbar_location = None
    abs_correlations = abs(correlations)
    unique_correlations, frequencies = numpy.unique(abs_correlations, return_counts=True)
    nan_indexes = numpy.isnan(unique_correlations)
    unique_correlations = unique_correlations[~nan_indexes]
    frequencies = frequencies[~nan_indexes]
    sorting_indexes = numpy.argsort(unique_correlations)
    unique_correlations, frequencies = unique_correlations[sorting_indexes], frequencies[sorting_indexes]

    plt = figure(title=f"",
                x_axis_label=label, y_axis_label=f"CDF",
                width=width, height=height)
    if unique_correlations.size > 2500000:
        step_size = max(unique_correlations.size // 10000000, 1)
        indexes = numpy.arange(0, unique_correlations.size, step_size)
        sampled_unique_correlations = unique_correlations[indexes]
        sampled_normalized_cumulative_sums = numpy.cumsum(frequencies[indexes]) / frequencies[indexes].sum()
    else:
        sampled_unique_correlations = unique_correlations
        sampled_normalized_cumulative_sums = numpy.cumsum(frequencies) / frequencies.sum()

    sampled_unique_correlations = numpy.hstack((sampled_unique_correlations, numpy.ones(1,)))
    sampled_normalized_cumulative_sums = numpy.hstack((sampled_normalized_cumulative_sums, numpy.ones(1,)))
    sampled_unique_correlations = numpy.hstack((numpy.zeros(1,), sampled_unique_correlations))
    sampled_normalized_cumulative_sums = numpy.hstack((numpy.zeros(1,), sampled_normalized_cumulative_sums))

    df = ColumnDataSource(data={
        "x": sampled_unique_correlations,
        "y": sampled_normalized_cumulative_sums
    })

    plt.line([0, 1], [1, 1], line_dash="dashed", color="black", line_width=3)
    # plt.line(sampled_unique_correlations, sampled_normalized_cumulative_sums, color="red", line_width=3)
    plt.line(sampled_unique_correlations, sampled_normalized_cumulative_sums, color="red", line_width=3)
    plt.varea(source=df, y1=0, y2="y", x="x", fill_alpha=0.2, fill_color="red")
    plt.x_range = Range1d(0., 1.)
    plt.y_range = Range1d(0., 1.01)
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "30pt"
    plt.xaxis.major_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    plt.yaxis.axis_label_text_font_size = "30pt"
    plt.title.text_font_size = "56pt"
    
    return plt


def draw_correlations_pdf(correlations, width=900, height=450):   
    plt = figure(title=f"",
                 x_axis_label="| ρ |", y_axis_label=f"PDF",
                 width=width, height=height,
                 toolbar_location=None)
    # plt.toolbar_location = None
    abs_correlations = abs(correlations)
    unique_correlations, frequencies = numpy.unique(abs_correlations, return_counts=True)
    nan_indexes = numpy.isnan(unique_correlations)
    unique_correlations = unique_correlations[~nan_indexes]
    frequencies = frequencies[~nan_indexes]
    frequencies = numpy.cumsum(frequencies)
    sorting_indexes = numpy.argsort(unique_correlations)
    unique_correlations, frequencies = unique_correlations[sorting_indexes], frequencies[sorting_indexes]

    plt = figure(title=f"",
                x_axis_label="Correlation", y_axis_label=f"CDF",
                width=width, height=height)
    if unique_correlations.size > 2500000:
        step_size = max(unique_correlations.size // 10000000, 1)
        indexes = numpy.arange(0, unique_correlations.size, step_size)
        sampled_unique_correlations = unique_correlations[indexes]
        sampled_normalized_cumulative_sums = frequencies[indexes] / frequencies[indexes].sum()
    else:
        sampled_unique_correlations = unique_correlations
        
        sampled_normalized_cumulative_sums = frequencies / frequencies.sum()

    sampled_unique_correlations = numpy.hstack((sampled_unique_correlations, numpy.ones(1,)))
    sampled_normalized_cumulative_sums = numpy.hstack((sampled_normalized_cumulative_sums, numpy.ones(1,)))
    sampled_unique_correlations = numpy.hstack((numpy.zeros(1,), sampled_unique_correlations))
    sampled_normalized_cumulative_sums = numpy.hstack((numpy.zeros(1,), sampled_normalized_cumulative_sums))

    df = ColumnDataSource(data={
        "x": sampled_unique_correlations,
        "y": sampled_normalized_cumulative_sums
    })

    plt.line([0, 1], [1, 1], line_dash="dashed", color="black", line_width=3)
    # plt.line(sampled_unique_correlations, sampled_normalized_cumulative_sums, color="red", line_width=3)
    plt.line(sampled_unique_correlations, sampled_normalized_cumulative_sums, color="red", line_width=3, width=40)
    # plt.varea(source=df, y1=0, y2="y", x="x", fill_alpha=0.2, fill_color="red")
    plt.x_range = Range1d(0., 1.)
    # plt.y_range = Range1d(0., 1.01)
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "30pt"
    plt.xaxis.major_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    plt.yaxis.axis_label_text_font_size = "30pt"
    plt.title.text_font_size = "56pt"
    
    return plt


def mean_nonzero_coefficients_per_correlation(correlations, y_univariates, y_multivariates, width=900, height=450):
    colors = ["blue", "purple", "green", "yellow", "orange", "red"]
    markers = ["triangle", "square", "star", "hex", "circle"]
    
    plt = figure(title=f"Absolute node count difference on increasing correlation",
                 x_axis_label="Correlation", y_axis_label="Prop. Coeff. != 0",
                 frame_width=width, frame_height=height,
                 toolbar_location=None)
    plt.x_range = Range1d(0.1, 0.9)

    min_y, max_y = numpy.inf, -numpy.inf
    legend_it = list()
    for i, (y_univariate, y_multivariate, c, marker) in enumerate(zip(y_univariates, y_multivariates, colors, markers)):
        y = y_univariate - y_multivariate
        plt.line(correlations, y, line_width=10, color=c, line_dash="dashed")
        # legend_it.append(("𝜖 = " + str(i / 10), [line]))

        min_y = min_y if y.min() > min_y else y.min()
        max_y = max_y if y.max() < max_y else y.max()
   
    for i, (y_univariate, y_multivariate, c, marker) in enumerate(zip(y_univariates, y_multivariates, colors, markers)):
        y = y_univariate - y_multivariate
        plt.scatter(correlations, y, marker=marker, color=c, size=100, fill_alpha=.5)
    # plt.add_layout(legend, 'right')

    plt.y_range = Range1d(min_y - min_y * 0.1, max_y + max_y * 0.1)
    plt.x_range = Range1d(0.1, 0.9)
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "80pt"
    plt.xaxis.major_label_text_font_size = "80pt"
    plt.xaxis.axis_label_text_font_size = "80pt"
    plt.xaxis.axis_label_text_font_size = "0pt"
    plt.yaxis.axis_label_text_font_size = "80pt"
    plt.title.text_font_size = "0pt"
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].axis_line_width = 5
    plt.yaxis[0].axis_line_width = 5
    plt.legend.title_text_font_size = "80pt"
    # plt.legend.label_text_font_size = "80pt"

    
       
    return plt


def draw_dimensionality_and_size(names, dataset_sizes, dataset_dimensionalities, dataset_numeric_dimensionalities, width=900, height=450):
    data_source = pandas.DataFrame.from_dict({
        "size": dataset_sizes,
        "dimensionality": dataset_dimensionalities,
        "numeric_dimensionality": dataset_numeric_dimensionalities,
        "name": names    
    })

    source_by_dimensionality = data_source.sort_values(by="size")
    hover = [
        ("Rank", "$index"),
        ("Size", "@size"),
        ("Dimensionality", "@dimensionality"),
        ("Dataset","@name")
    ]
    
    plt = figure(title="Dataset sizes", tooltips=hover,
                 width=width, height=height)
    plt.yaxis.axis_label = "Dimensionality"
    plt.xaxis.axis_label = "Size"
    plt.x_range = Range1d(-100, max(dataset_sizes) * 1.1)

    plt.scatter(x="size", y="dimensionality", source=ColumnDataSource(source_by_dimensionality),
                size=20, color="blue", fill_alpha=0.5, line_alpha=0.5,
                legend_label="Dimensionality")
    plt.scatter(x="size", y="numeric_dimensionality", source=ColumnDataSource(source_by_dimensionality),
                size=20, color="yellow", fill_alpha=0.5, line_alpha=0.5,
                legend_label="Numeric-only dimensionality")
    
    plt.legend.orientation = "vertical"
    plt.legend.location = "top_left"
    plt.legend.click_policy = "hide"
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "15pt"
    plt.xaxis.major_label_text_font_size = "15pt"
    plt.xaxis.axis_label_text_font_size = "15pt"
    plt.yaxis.axis_label_text_font_size = "15pt"
    plt.title.text_font_size = "28pt"


    return plt


def trees_performance_differences(differences, metrics, width = 900, height = 450, marker="triangle"):
    plt = figure(title=f"UDT vs MDT",
                 x_axis_label="", y_axis_label=f"Gap",
                 frame_width=width, frame_height=height,
                 toolbar_location=None)
    x = numpy.arange(differences.size)
    max_y = differences.max()
    min_y = differences.min()
    plt.x_range = Range1d(0, x.size)
    plt.y_range = Range1d(-40, 40)

    y_mean = differences.mean()
    y_std = differences.std()
    mean_y = numpy.array([y_mean] * x.size)
    df = pandas.DataFrame({
        metrics: differences
    })
    df["x"] = x
    df["mean_y"] = mean_y
    df["upper_y"] = mean_y + y_std
    df["lower_y"] = mean_y - y_std
    df["min_y"] = [min_y] * x.size
    df["max_y"] = [max_y * 1.1] * x.size
    
    source = ColumnDataSource(df)
    
    plt.line([0, x.size - 1], [y_mean, y_mean], line_dash="dashed", color="purple", line_width=10, line_alpha=0.5)
    plt.line([0, x.size - 1], [y_mean + y_std, y_mean + y_std], color="red", line_width=10, line_alpha=.5)
    plt.line([0, x.size - 1], [y_mean - y_std, y_mean - y_std], color="blue", line_width=10, line_alpha=.5)

    univariate_label = Label(x=1, y=max_y - 5, text="UDTs >> MDTs", text_font_size="60pt", text_color="red", text_font="CMU")
    multivariate_label = Label(x=x.size - 20, y=min_y + 1, text="MDTs >> UDTs", text_font_size="60pt", text_color="blue", text_font="CMU")
    plt.add_layout(univariate_label)
    plt.add_layout(multivariate_label)

    band = Band(base="x", lower="lower_y", upper="upper_y", source=source,
                fill_alpha=0.1, fill_color="purple", line_color="purple")
    univariate_band = Band(base="x", lower=-40, upper="lower_y", source=source,
                           fill_alpha=0.1, fill_color="blue", line_color="blue")
    multivariate_band = Band(base="x", lower="upper_y", upper=40, source=source,
                             fill_alpha=0.1, fill_color="red", line_color="red")
    plt.add_layout(band)
    plt.add_layout(univariate_band)
    plt.add_layout(multivariate_band)

    colors = ["blue", "green", "yellow", "red"]
    markers = ["triangle", "square", "star", "hex", "circle"]
    
    differences = numpy.sort(differences)
    indexes_of_univariate_outliers = numpy.argwhere(differences >= y_mean + y_std).squeeze()
    indexes_of_multivariate_outliers = numpy.argwhere(differences <= y_mean - y_std).squeeze()
    indexes_of_others = [i for i in range(x.size) if i not in indexes_of_multivariate_outliers and x not in indexes_of_univariate_outliers]
    for c, indexes in zip(["red", "blue", "purple"], [indexes_of_univariate_outliers, indexes_of_multivariate_outliers, indexes_of_others]):
        filtered_y = differences[indexes]
        filtered_y = numpy.sort(filtered_y)
        plt.scatter(indexes, filtered_y, color=c,
                    # legend_label=metric,
                    marker=marker, fill_alpha=0.5, size=50)

    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    # plt.yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    # plt.legend.orientation = "horizontal"
    # plt.legend.location = "bottom_left"
    # plt.legend.click_policy = "hide"
    # plt.legend.location = "bottom_left"
    plt.yaxis.major_label_text_font_size = "60pt"
    plt.xaxis.major_label_text_font_size = "60pt"
    plt.xaxis.axis_label_text_font_size = "60pt"
    plt.xaxis.axis_label_text_font_size = "0pt"
    plt.yaxis.axis_label_text_font_size = "60pt"
    plt.title.text_font_size = "0pt"
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].axis_line_width = 5
    plt.yaxis[0].axis_line_width = 5
    # plt.legend.title_text_font_size = "20pt"
    # plt.legend.label_text_font_size = "20pt"
    # plt.legend.orientation = "horizontal"
    # plt.legend.location = "top_left"    
       
    return plt


def draw_correlation_by_slope(correlations, slopes, width=900, height=480):
    plt = figure(
        x_axis_label = "Correlation",
        y_axis_label = "Slope",
        frame_width=width, frame_height=height,
        toolbar_location=None
    )

    for c, s in zip(correlations, slopes):
        plt.scatter(c, s, color="red", size=10, marker="circle", fill_alpha=0.1)
    
    plt.xaxis.axis_label_text_font = "CMU"
    plt.yaxis.axis_label_text_font = "CMU"
    plt.title.text_font = "CMU"
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None
    plt.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    plt.yaxis.major_label_text_font_size = "30pt"
    plt.xaxis.major_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "30pt"
    plt.xaxis.axis_label_text_font_size = "0pt"
    plt.yaxis.axis_label_text_font_size = "30pt"
    plt.title.text_font_size = "0pt"
    plt.yaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].ticker.desired_num_ticks = 5
    plt.xaxis[0].axis_line_width = 5
    plt.yaxis[0].axis_line_width = 5

    return plt