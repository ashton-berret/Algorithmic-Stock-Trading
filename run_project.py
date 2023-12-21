from vizro import Vizro
import vizro.models as vm
import pandas as pd
from typing import Callable
from typing_extensions import Literal
from dash import Input, Output, State, callback, callback_context, dcc, html, dash_table
from strategy_plot import *
"""
Need to update toolkit in slider to display date range
"""
class TooltipNonCrossRangeSlider(vm.RangeSlider):
    """Custom numeric multi-selector `TooltipNonCrossRangeSlider`."""

    type: Literal["other_range_slider"] = "other_range_slider"

    def build(self):
        value = self.value or [self.min, self.max]  # type: ignore[list-item]

        output = [
            Output(f"{self.id}_start_value", "value"),
            Output(f"{self.id}_end_value", "value"),
            Output(self.id, "value"),
            Output(f"temp-store-range_slider-{self.id}", "data"),
        ]
        input = [
            Input(f"{self.id}_start_value", "value"),
            Input(f"{self.id}_end_value", "value"),
            Input(self.id, "value"),
            State(f"temp-store-range_slider-{self.id}", "data"),
        ]

        @callback(output=output, inputs=input)
        def update_slider_values(start, end, slider, input_store):
            trigger_id = callback_context.triggered_id
            if trigger_id == f"{self.id}_start_value" or trigger_id == f"{self.id}_end_value":
                start_text_value, end_text_value = start, end
            elif trigger_id == self.id:
                start_text_value, end_text_value = slider
            else:
                start_text_value, end_text_value = input_store if input_store is not None else value

            start_value = min(start_text_value, end_text_value)
            end_value = max(start_text_value, end_text_value)
            start_value = max(self.min, start_value)
            end_value = min(self.max, end_value)

            slider_value = [start_value, end_value]

            start_date = pd.to_datetime(
                start_value, unit='s').strftime('%Y-%m-%d')
            end_date = pd.to_datetime(end_value, unit='s').strftime('%Y-%m-%d')

            return start_date, end_date, slider_value, (start_value, end_value)

        return html.Div(
            [
                html.P(self.title, id="range_slider_title") if self.title else None,
                html.Div(
                    [
                        dcc.RangeSlider(
                            id=self.id,
                            min=self.min,
                            max=self.max,
                            step=self.step,
                            # marks={},
                            marks=self.marks,
                            className="range_slider_control" if self.step else "range_slider_control_no_space",
                            value=value,
                            persistence=True,
                            allowCross=False,
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id=f"{self.id}_start_value",
                                    type='text',
                                    placeholder="start",
                                    readOnly=True,
                                    className="slider_input_field_left"
                                    if self.step
                                    else "slider_input_field_no_space_left",
                                    value="",
                                    size="100px",
                                    persistence=True,
                                ),
                                dcc.Input(
                                    id=f"{self.id}_end_value",
                                    type='text',
                                    placeholder="end",
                                    readOnly=True,
                                    className="slider_input_field_right"
                                    if self.step
                                    else "slider_input_field_no_space_right",
                                    value="",
                                    size="100px",
                                    persistence=True,
                                ),
                                dcc.Store(
                                    id=f"temp-store-range_slider-{self.id}", storage_type="local"),
                            ],
                            className="slider_input_container",
                        ),
                    ],
                    className="range_slider_inner_container",
                ),
            ],
            className="selector_container",
        )

df = pd.read_csv(
  "/Users/rileyoest/VS_Code/CSC-4444/Stock_Data/stock_data.csv"
  #"/Users/rileyoest/VS_Code/CSC-4444/Stock_Data/AAPL.csv"
)

df["DateTime"] = pd.to_datetime(df["Date"]).astype(
    int) / 10**9  # Convert to Unix timestamp


def select_range(df, start, end, symbol=None):
    if symbol:
        df = df[df['Symbol'] == symbol]
    return df[(df["Date"] >= start) & (df["Date"] <= end)]


dfplot = select_range(df, df['Date'].min(), df['Date'].max())

vm.Filter.add_type("selector", TooltipNonCrossRangeSlider)

main_page = vm.Page(
    title="Without Fetures",
    layout=vm.Layout(grid=[[i] for i in range(11)],  # range(len(components))
                     row_min_height="350px"),
    components=[
        vm.Card(
            text="""
            # Add Features

            * 30 day volatility
            * 30 day momentum
            * 5 day volatility
            * 5 day momentum
            * tweets sentiments
            * MACD
            * RSI
            * SMA
            * EMA
            """,
            href="/with-features",
        ),
        vm.Card(
            text="""
            # Linear Regression
            """, 
        ),
        vm.Graph(
            id="linear",
            figure=plot_linear_regression(dfplot),
        ), 
        vm.Card(
            text="""
            # Lasso Regression
            """, 
        ),
        vm.Graph(
            id="lasso",
            figure=plot_lasso_regression(dfplot)
        ),
        vm.Card(
            text="""
            # Ridge Regression
            """, 
        ),
        vm.Graph(
            id="ridge",
            figure=plot_ridge_regression(dfplot)
        ),
        vm.Card(
            text="""
            # Ridge/Lasso Combination Regression Model

            * Ridge Regression is used to predict High and Close

            * Lasso Regression is used to predict Low and Open
            """
        ),
        vm.Graph(
            id="combo",
            figure=plot_combo_regression(dfplot)
        ),
        vm.Card(
            text="""
            # Accuracy of Predictions on Testing Set
            """
        ),
        vm.Graph(
            id="metrics",
            figure=plot_metrics(dfplot) 
        ),
       
    ],
    controls=[
        # Change Symbol Filter to only be able to select one stock
        vm.Filter(column="Symbol", targets=['linear', 'lasso', 'ridge', 'combo', 'metrics']),  # Dropdown filter for the Symbol column
        vm.Filter(
            column="DateTime",
            targets=['linear', 'lasso', 'ridge', 'combo', 'metrics'],
            selector=TooltipNonCrossRangeSlider(),
        ),
    ],
)

add_features_page = vm.Page(
    title="With Features",
    layout=vm.Layout(grid=[[i] for i in range(11)],  # range(len(components))
                     row_min_height="350px"),
    components=[
        vm.Card(
            text="""
            # Remove Features
            """,
            href="/without-features",
        ),
        vm.Card(
            text="""
            # Linear Regression
            """, 
        ),
        vm.Graph(
            id="linear_features",
            figure=plot_linear_regression_with_features(dfplot),
        ), 
        vm.Card(
            text="""
            # Lasso Regression
            """, 
        ),
        vm.Graph(
            id="lasso_features",
            figure=plot_lasso_regression_with_features(dfplot)
        ),
        vm.Card(
            text="""
            # Ridge Regression
            """, 
        ),
        vm.Graph(
            id="ridge_features",
            figure=plot_ridge_regression_with_features(dfplot)
        ),
        vm.Card(
            text="""
            # Ridge/Lasso Combination Regression Model

            * Ridge Regression is used to predict High and Close

            * Lasso Regression is used to predict Low and Open
            """
        ),
        vm.Graph(
            id="combo_features",
            figure=plot_combo_regression_with_features(dfplot)
        ),
        vm.Card(
            text="""
            # Accuracy of Predictions on Testing Set
            """
        ),
        vm.Graph(
            id="metrics_features",
            figure=plot_metrics(dfplot) 
        ),
       
    ],
    controls=[
        # Change Symbol Filter to only be able to select one stock
        vm.Filter(column="Symbol", targets=['linear_features', 'lasso_features', 'ridge_features',
                  'combo_features', 'metrics_features']),  # Dropdown filter for the Symbol column
        vm.Filter(
            column="DateTime",
            targets=['linear_features', 'lasso_features', 'ridge_features', 'combo_features', 'metrics_features'],
            selector=TooltipNonCrossRangeSlider(),
        ),
    ],
)

dashboard = vm.Dashboard(pages=[main_page, add_features_page])

def main():
    Vizro().build(dashboard).run()
    
if __name__ == "__main__":
    main()
