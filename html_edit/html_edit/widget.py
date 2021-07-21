from ipywidgets import DOMWidget, ValueWidget, register
from traitlets import Unicode, Bool, List, Tuple, validate, TraitError

from ._frontend import module_name, module_version


@register
class customWidget(DOMWidget, ValueWidget):
    _model_name = Unicode('customModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    continuous_update = Bool(True, help="Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.").tag(sync=True)

    _view_name = Unicode('customView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)
    #continuous_update = Bool(True, help="Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.").tag(sync=True)

    value = Unicode("Hello, world!").tag(sync=True)
    
@register
class customWidget2(DOMWidget, ValueWidget):
    _model_name = Unicode('customModel2').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    continuous_update = Bool(True, help="Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.").tag(sync=True)

    _view_name = Unicode('customView2').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)
    #continuous_update = Bool(True, help="Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.").tag(sync=True)

    value = Bool(False).tag(sync=True)

@register
class customWidget3(DOMWidget, ValueWidget):
    _model_name = Unicode('customModel3').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    continuous_update = Bool(True, help="Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.").tag(sync=True)

    _view_name = Unicode('customView3').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)
    #continuous_update = Bool(True, help="Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.").tag(sync=True)

    #value = Unicode("0").tag(sync=True)
    values = List([0,0,0,0,0,0,0,0,0,0]).tag(sync=True)