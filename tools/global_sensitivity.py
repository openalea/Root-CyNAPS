import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score

global_state_extracts = dict(
    total_Nm=dict(unit="mol", value_example="not provided",  description="not provided"),
    total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    total_hexose=dict(unit="mol", value_example="not provided", description="not provided"),
    #total_cytokinins=dict(unit="mol", value_example="not provided", description="not provided"),
    total_struct_mass=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_Nm=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    phloem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    #xylem_total_water=dict(unit="mol", value_example="not provided", description="not provided"),
    #xylem_total_pressure=dict(unit="Pa", value_example="not provided", description="not provided")
)

global_flow_extracts = dict(
    Nm_root_shoot_xylem=dict(unit="mol.time_step-1", value_example="not provided",  description="not provided"),
    AA_root_shoot_xylem=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    #Unloading_Amino_Acids=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    #Export_cytokinins=dict(unit="UA.time_step-1", value_example="not provided", description="not provided"),
    cytokinin_synthesis=dict(unit="mol", value_example="not provided", description="not provided"),
    #actual_transpiration=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    #Total_Transpiration=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    total_AA_rhizodeposition=dict(unit="mol.time_step-1", value_example="not provided", description="not provided")
)


def regression_analysis(dataset, output_path):
    # TODO : normalize and comment
    regression_variables = [dim for dim in dataset.dims if dim not in ('t', 'vid')]
    df_regression = pd.DataFrame(columns=['variable', 'r2', 'intercept'] + regression_variables)
    for global_output in global_state_extracts.keys():
        y = dataset.sel(vid=1)[global_output]
        y = y.sel(t=max(y.t))
        y = y.stack(stk=y.dims)
        y = (y - y.min()) / (y.max() - y.min())
        y = y.fillna(0)
        x = y.coords['stk'].to_numpy()
        x = [list(k) for k in x]
        y = list(y.to_numpy())

        regressor = LinearRegression()
        regressor.fit([list(k) for k in x], y)
        y_pred = regressor.predict(x)
        r2 = r2_score(y, y_pred)

        keys = ['variable', 'r2', 'intercept'] + regression_variables
        values = [global_output, r2, regressor.intercept_] + [coef for coef in regressor.coef_]

        df_regression.loc[len(df_regression)] = dict(zip(keys, values))

    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table = pd.plotting.table(ax, df_regression, loc='upper right')
    table.auto_set_font_size(True)
    plt.savefig(output_path + '/linear_regression.png', dpi=300)

    return