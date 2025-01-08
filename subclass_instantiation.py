from typing import Any, TypeVar, Type


def parse_kwargs_str(kwargs_str: str) -> dict[str, Any]:
    '''Parse a kwargs_str of the form "arg1=val1, arg2=val2, ..."\n
    As in Leetcode 1106, parsing the kwargs isn't as simple as just
    .split(,)...I have to make sure prior brackets are closed

    :param kwargs_str:
    :return: dict keyed by argument names {arg1:val1, arg2:val2, ...}
    '''
    kwarg_strings: list[str] = []
    cur_kwarg_chars: list[str] = []
    net_opens = {'(': 0, '[': 0, '{': 0}
    opens_by_closes = {')': '(', ']': '[', '}': '{'}
    for ch in kwargs_str:
        if ch == ',' and all([nv == 0 for nv in net_opens.values()]):
            kwarg_strings.append(''.join(cur_kwarg_chars))
            cur_kwarg_chars = []
            continue
        if ch in net_opens:
            net_opens[ch] += 1
        if ch in opens_by_closes:
            net_opens[opens_by_closes[ch]] -= 1
        cur_kwarg_chars.append(ch)
    kwarg_strings.append(''.join(cur_kwarg_chars))
    cur_kwarg_chars = []

    kwarg_dict: dict[str, Any] = {}
    for kwarg_str in kwarg_strings:
        equals_index = kwarg_str.index('=')
        arg_name = kwarg_str[:equals_index]
        arg_value = eval(kwarg_str[equals_index+1:])
        kwarg_dict[arg_name] = arg_value
    return kwarg_dict


def parse_class_name_kwargs(
        class_name_optional_kwargs: str,
        print_debugging: bool = False) -> tuple[str, dict[str, Any]]:
    '''Given the argument from the command line, of the form\n
    ClassName(a=1, b=2, ...), parse it into class name and kwargs

    :param class_name_optional_kwargs: of the form ClassName(a=1, b=2, ...)
        Note that some kwargs could be lists, tuples, even dicts
    :param print_debugging: if True, print the parsed kwarg dict
        (for debugging purposes)
    :return: class_name, kwargs dict
    '''
    if '(' not in class_name_optional_kwargs:
        return class_name_optional_kwargs, {}

    first_paren_index = class_name_optional_kwargs.index('(')
    class_name = class_name_optional_kwargs[:first_paren_index]
    all_kwargs_str = class_name_optional_kwargs[first_paren_index+1:-1]
    kwarg_dict = parse_kwargs_str(all_kwargs_str)

    if print_debugging:
        print(f'After parsing: {class_name=}, {kwarg_dict=}')
    return class_name, kwarg_dict


def recursive_get_all_subclasses(base_class_type: type) -> list[type]:
    '''The __subclasses__() method only returns "immediate subclasses"
    This recursively gets all child classes of a base class'''
    all_subclasses: list[type] = []
    for subclass in base_class_type.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(recursive_get_all_subclasses(subclass))
    return all_subclasses


T = TypeVar('T', bound=object)


def get_class_instance(class_name_optional_kwargs: str,
                       base_class_type: Type[T],
                       print_debugging: bool = False,
                       **other_kwargs: Any) -> T:
    '''Instantiate a subclass of a given abstract base class\n
    Note that imports need to be structured properly for this to work\n
    Helpful: https://peps.python.org/pep-0484/#the-type-of-class-objects
        This section in PEP 484 explains the type hinting

    :param class_name_optional_kwargs: Instead of just passing class name,
        might be more convenient to allow arguments of the form:
        "ClassName(arg1=val1, ..., argn=valn)" (see pg 830)
    :param base_class_type: type of the base class
    :param print_debugging: if True, print the names of the subclasses
        that were found (for debugging purposes)
    :param other_kwargs: other arguments passed into the subclass
        instantiation (not including the ones imbedded in the
        above class_name_optional_kwargs above)
    :return: an instantiated instance of the correct class
    '''
    possible_base_subclasses: dict[str, type] = {
        cls.__name__: cls
        for cls in recursive_get_all_subclasses(base_class_type)}
    class_name, embedded_kwargs = parse_class_name_kwargs(
        class_name_optional_kwargs, print_debugging)
    all_kwargs = embedded_kwargs | other_kwargs
    if print_debugging:
        print(f'\n{base_class_type=}, {possible_base_subclasses=}, '
              f'{all_kwargs=}')
    return possible_base_subclasses[class_name](**all_kwargs)
