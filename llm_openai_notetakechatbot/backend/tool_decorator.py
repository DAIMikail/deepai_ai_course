# Python fonksiyonlarını OpenAI tool şemasına çeviren decorator

import inspect
from typing import Callable, get_type_hints


def openai_tool(func: Callable) -> Callable:
    """
    Python fonksiyonunu OpenAI tool şemasına çevirir.
    Fonksiyon çalışmaya devam eder, şemaya func.schema ile erişilir.

    Kullanım:
        @openai_tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Belirtilen şehrin hava durumunu getirir.

            Args:
                city: Hava durumu sorgulanacak şehir adı
                unit: Sıcaklık birimi
            '''
            return f"{city}: 20°C"

        # Fonksiyonu çağır
        print(get_weather("İstanbul"))

        # Tool şemasına eriş
        tools = [get_weather.schema]
    """
    name = func.__name__
    doc = func.__doc__ or ""
    description = doc.split("\n")[0].strip() if doc else f"{name} fonksiyonu"

    # Type hints ve parametreleri al
    type_hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    sig = inspect.signature(func)
    param_descriptions = _parse_docstring_params(doc)

    # Properties ve required listesini oluştur
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        json_type = _python_type_to_json(param_type)

        properties[param_name] = {
            "type": json_type,
            "description": param_descriptions.get(param_name, f"{param_name} parametresi")
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Şemayı fonksiyona attribute olarak ekle
    func.schema = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

    return func


def _python_type_to_json(python_type) -> str:
    """Python tipini JSON Schema tipine çevirir."""
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    return type_mapping.get(python_type, "string")


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Docstring'den parametre açıklamalarını çıkarır."""
    params = {}
    if not docstring:
        return params

    lines = docstring.split("\n")
    in_args = False

    for line in lines:
        line = line.strip()
        if line.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if line.startswith("Returns:") or line.startswith("Raises:") or not line:
                if not line:
                    continue
                break
            if ":" in line:
                parts = line.split(":", 1)
                param_name = parts[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                params[param_name] = param_desc

    return params
