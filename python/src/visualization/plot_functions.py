import plotly.graph_objects as go
from plotly.offline import iplot
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

def paper_boxplot_comparison(df, filename, colors=['#59C3C3', '#FDA96D'], showlegend=True):
    traces = []
    for i, md in enumerate(df.clf.unique()):
        df_plot = df[(df.clf==md)]
        trace = go.Box(
                    x=df_plot.metric, 
                    y=df_plot.score,
                    boxpoints='outliers',
                    name=md,
                    fillcolor=colors[i],
                    line_color = '#000000',
                    line_width=3,
                    marker=dict(size=20, color=colors[i])
                    )
        traces.append(trace)
    fig = go.Figure()
    fig.add_traces(traces)
    fig.update_layout(
        boxmode="group",
        yaxis_title_text=df.quality_metric.unique()[0].title(),
        height=1000,
        width=2000,
        title_x=0.5,
        legend=dict(font=dict(size=60, color='black'), yanchor="bottom", y=0.075, xanchor="right", x=0.99),
        xaxis = dict(tickfont = dict(size=60), showline=True, linewidth=3, linecolor='black', mirror=True, color='black'),
        yaxis = dict(tickfont = dict(size=60), titlefont=dict(size=60), title_standoff = 30, showline=True, linewidth=3, linecolor='black', mirror=True, color='black'),
        boxgroupgap=0.2,
        boxgap=0.1,
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        margin=dict(l=20, r=20, t=25, b=20),
        showlegend=showlegend)
    fig.show()
    fig.write_image('../../../python/reports/plots/{}.jpeg'.format(filename))


def paper_boxplot_comparison_single_metric(df, filename, colors=['#FDA96D', '#59C3C3'], showlegend=True):
    warnings.filterwarnings('ignore')
    traces = []
    for i, md in enumerate(df.clf.unique()):
        order_df = pd.DataFrame({'method':['Process', 'SCA', 'Conservative']})
        
        order_df = order_df.reset_index().set_index('method')
        df_plot = df[(df.clf==md)]
        df_plot['method_order'] = df_plot['method'].map(order_df['index'])
        df_plot.sort_values(['quality_metric', 'method_order'], inplace=True)
        trace = go.Box(
                    x=[df_plot.method, df_plot.quality_metric], 
                    y=df_plot.score,
                    boxpoints='outliers',
                    name=md,
                    fillcolor=colors[i],
                    line_color = '#000000',
                    line_width=3,
                    marker=dict(size=20, color=colors[i])
                    )
        traces.append(trace)
    fig = go.Figure()
    fig.add_traces(traces)
    fig.update_layout(
        boxmode="group",
        yaxis_title_text='Score',
        height=1000,
        width=2000,
        title_x=0.5,
        legend=dict(font=dict(size=60, color='black'), yanchor="bottom", y=1.01, xanchor="right", x=0.99, orientation='h'),
        xaxis = dict(tickfont = dict(size=60), showline=True, linewidth=3, linecolor='black', mirror=True, color='black'),
        yaxis = dict(tickfont = dict(size=60), titlefont=dict(size=60), title_standoff = 30, showline=True, linewidth=3, linecolor='black', mirror=True, color='black'),
        boxgroupgap=0.2,
        boxgap=0.1,
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        margin=dict(l=20, r=20, t=25, b=20),
        showlegend=showlegend)
    fig.write_image('../../../python/reports/plots/{}.jpeg'.format(filename))
    fig.show()