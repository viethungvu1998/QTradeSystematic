# Trial 001_baseline_import_fix

- Status: crash
- Stop reason: continue
- Objective: sharpe = None
- Best value: None
- Criteria hit: false
- Description: baseline after vectorbt import compatibility fix

## Log Tail

```text
Traceback (most recent call last):
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 35, in <module>
    import vectorbtpro as vbt
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/__init__.py", line 30, in <module>
    from vectorbtpro._settings import settings
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/_settings.py", line 2288, in <module>
    settings.reset_theme()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/_settings.py", line 2260, in reset_theme
    self.set_theme(self["plotting"]["default_theme"])
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/_settings.py", line 2254, in set_theme
    self.register_template(theme)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/_settings.py", line 2245, in register_template
    pio.templates["vbt_" + theme] = go.layout.Template(template)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/plotly/io/_templates.py", line 94, in __setitem__
    self._templates[key] = self._validate(value)
                           ~~~~~~~~~~~~~~^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/plotly/io/_templates.py", line 110, in _validate
    return self._validator.validate_coerce(value)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/_plotly_utils/basevalidators.py", line 2746, in validate_coerce
    if v == {} or isinstance(v, self.data_class) and v.to_plotly_json() == {}:
                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```
