// Copyright (c) Jeremy Alcanzare
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
} from '@jupyter-widgets/base';

import { MODULE_NAME, MODULE_VERSION } from './version';

// Import the CSS
import '../css/widget.css';

export class customModel extends DOMWidgetModel {
  defaults() {
    return {...super.defaults(),
      _model_name: customModel.model_name,
      _model_module: customModel.model_module,
      _model_module_version: customModel.model_module_version,
      _view_name: customModel.view_name,
      _view_module: customModel.view_module,
      _view_module_version: customModel.view_module_version,
      disabled: false,
      continuous_update: true
    };
  }

  static serializers: ISerializers = {
      ...DOMWidgetModel.serializers,
      // Add any extra serializers here
    }

  static model_name = 'customModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'customView';
  static view_module = MODULE_NAME;
  static view_module_version = MODULE_VERSION;
}

export class customModel2 extends DOMWidgetModel {
  defaults() {
    return {...super.defaults(),
      _model_name: customModel2.model_name,
      _model_module: customModel2.model_module,
      _model_module_version: customModel2.model_module_version,
      _view_name: customModel2.view_name,
      _view_module: customModel2.view_module,
      _view_module_version: customModel2.view_module_version,
      value:false,
      disabled: false,
      continuous_update: true
    };
  }

  static serializers: ISerializers = {
      ...DOMWidgetModel.serializers,
      // Add any extra serializers here
    }

  static model_name = 'customModel2';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'customView2';
  static view_module = MODULE_NAME;
  static view_module_version = MODULE_VERSION;
}

export class customModel3 extends DOMWidgetModel {
  defaults() {
    return {...super.defaults(),
      _model_name: customModel3.model_name,
      _model_module: customModel3.model_module,
      _model_module_version: customModel3.model_module_version,
      _view_name: customModel3.view_name,
      _view_module: customModel3.view_module,
      _view_module_version: customModel3.view_module_version,
      values: [0,0,0,0,0,0,0,0,0,0],
      disabled: false,
      continuous_update: true
    };
  }

  static serializers: ISerializers = {
      ...DOMWidgetModel.serializers,
      // Add any extra serializers here
    }

  static model_name = 'customModel3';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'customView3';
  static view_module = MODULE_NAME;
  static view_module_version = MODULE_VERSION;
}




export class customView extends DOMWidgetView {
   render() {
    this._htmlInput = document.createElement('input');
    this._htmlInput.type = 'search';
    this._htmlInput.value = this.model.get('value');
    this._htmlInput.disabled = this.model.get('disabled');

    this.el.appendChild(this._htmlInput);

    // Python -> JavaScript update
    this.model.on('change:value', this._onValueChanged, this);
    this.model.on('change:disabled', this._onDisabledChanged, this);

    // JavaScript -> Python update
    this._htmlInput.onchange = this._onInputChanged.bind(this);
  }

  private _onValueChanged() {
    this._htmlInput.value = this.model.get('value');
  }

  private _onDisabledChanged() {
    this._htmlInput.disabled = this.model.get('disabled');
  }

  private _onInputChanged() {
    this.model.set('value', this._htmlInput.value);
    this.model.save_changes();
  }

  private _htmlInput: HTMLInputElement;
}

export class customView2 extends DOMWidgetView {
   render() {
    this._htmlInput = document.createElement('input');
    this._htmlInput.className = 'checkboxInput';
    this._htmlInput.type = 'checkbox';
    this._htmlInput.checked = this.model.get('value');
    this._htmlInput.disabled = this.model.get('disabled');

    this.el.appendChild(this._htmlInput);
        
    // Python -> JavaScript update
    this.model.on('change:value', this._onValueChanged, this);
    this.model.on('change:disabled', this._onDisabledChanged, this);

    // JavaScript -> Python update
    this._htmlInput.onchange = this._onInputChanged.bind(this);
  }
    
  private _onValueChanged() {
      this._htmlInput.checked = this.model.get('value');
  }
    
  private _onDisabledChanged() {
    this._htmlInput.disabled = this.model.get('disabled');
  }

  private _onInputChanged() {
    this.model.set('value', this._htmlInput.checked);
    this.model.save_changes();
  }

  private _htmlInput: HTMLInputElement;
}

export class customView3 extends DOMWidgetView {
   render() {
    this.model.values = this.model.get('values');
    for(var i = 0; i<10; i++){
    this._htmlInput = document.createElement('input');
    this._htmlInput.className = 'checkboxInput';
    this._htmlInput.type = 'checkbox';
    this._htmlInput.id = i.toString();
    //this._htmlInput.checked = this.model.get('value');
    this._htmlInput.disabled = this.model.get('disabled');

    this.el.appendChild(this._htmlInput);
        
    // Python -> JavaScript update
    this.model.on('change:values', this._onValueChanged, this);
    this.model.on('change:disabled', this._onDisabledChanged, this);

    //this.model.on('change:values', this._onInputChanged, this);
  }
   }

  private _onValueChanged(e: Event) {
    console.log("value changed!");
    let values = this.model.get('values');
    var html_elements = document.getElementsByClassName('checkboxInput')
    let i = -1;
    for(let html_element of Array.from(html_elements)) 
       {
        const html_input = html_element as HTMLInputElement;
       //console.log(html_element.checked);
        html_input.checked = Boolean(values[i])
        //html_input.checked = this.model.get('value');
        i+=1;
       }
  }
    
  private _onDisabledChanged() {
    this._htmlInput.disabled = this.model.get('disabled');
  }

    events(): { [e: string]: string } {
    return {
      'click input[type="checkbox"]': '_handle_click',
    };
  }

  /**
   * Handles when the checkbox is clicked.
   *
   * Calling model.set will trigger all of the other views of the
   * model to update.
   */
  _handle_click(e: Event): void {
    console.log(e.target);
    //const html_input = e.target as HTMLInputElement
    console.log("Initial Array ")
    console.log(this.model.get('values'))
    let values = this.model.get('values');
    const html_input = e.target as HTMLInputElement;
        if(html_input.checked) 
        {
            values[Number(html_input.id)] = 1;
        }
        else {
            values[Number(html_input.id)] = 0;
        }
    //const value = this.model.get('values');
    console.log("Final Array ")
    console.log(this.model.get('values'))
    this.model.set('values', values);
    this.touch();
  }

  private _htmlInput: HTMLInputElement;
}
