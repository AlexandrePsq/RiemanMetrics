import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

%%
def plot(results, kind="bar", y="score", x="subject", hue="pipeline", palette="tab10", height=40):
    sns.set(font_scale=8)
    g = sns.catplot(
        kind=kind,
        y=y,
        x=x,
        hue=hue,
        data=results,
        #orient="h",
        palette=palette,
        height=height,
    )
    #g.set_xticklabels(plt.get_xticklabels(), fontsize = 18)
    plt.show()


df = pd.read_csv('results_light.csv')
for d in df['dataset'].unique():
    plot(df[df['dataset'] == d])
# plot(df)
