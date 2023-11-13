"""데이터셋에 따른 prompt 집합. 반드시 이 파일에 있는 prompt를 불러와서 사용해야 함. 하드코딩은 허용되지 않음.

FLAN처럼 디자인된 점:
    - instruction에 옵션이 주어짐
    - 'input' key는 사용하지 않음
    - 모든 instruction variation에 대해 output은 같음.
    - 지금은 데이터셋당 5개의 instruction밖에 가지고 있지 않으나, 10개로 늘려야 함.
FLAN과 다르게 디자인된 점:
    - FLAN은 모든 template에 대해 {options_}, {answer} key를 사용해서 option과 answer를 주도록 통일했으나,
        여기서는 데이터셋에 주어진 key를 활용해서 명시적으로 template을 만듦
{options}에 들어갈 문자열 양식:
    선택지:
     - options1
     - options2
     ...
"""
import copy

from korean_utils import bojosa


datasets = {
    # 필요한 key:
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "hellaswag": [
        {
            'instruction': '다음의 글을 읽고 물음에 답하세요.\n\n{context}\n\n글에서 이어질 문장으로 가장 올바른 것은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n이어질 말로 제일 적당한 문장을 골라 줘.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n에 가장 어울리는 다음 말은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n주어진 선택지 중 다음 단락에 이어지기에 가장 자연스러운 것은?\n{context}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n주어진 지문을 계속 쓴다면 다음에 올 말을 선택해.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 단락 이후에 어떤 일이 일어날까?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '글에 이어서 문장을 더 써줘.\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 다음 문장은?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '이 이야기가 어떻게 끝날까??\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{answer}\n이 문장 이전에 있을 법한 이야기를 써 줘.',
            'input': '',
            'output': '{context}'
        },
    ],
    # 필요한 key:
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "copa": [
        {
            'instruction': '다음 상황이 주어졌을 때, 이 상황의 {question}{euro_or_ro} 적절한 것을 고르시오.\n상황: {premise}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}\n위 사건의 {question}{eun_or_neun}?\n상황: {premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{premise}"가 일어나게 된 {question}{eun_or_neun}??\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 현상의 {question}{euro_or_ro} 더 적절한 것을 골라줘.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n\n둘 중에 {premise}의 {question}인 것은 무엇인가?',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}의 {question} 생성해\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '현상: {premise}\n{question}:\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 글을 읽고 물음에 답하시오.\n{premise}\n{question}{eun_or_neun} 무엇인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 선택지를 골라.\n"{premise}"의 {question}{eun_or_neun}?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{answer}"가 {question}인 사건을 생성해 줘.',
            'input': '',
            'output': '{premise}'
        },
    ],
    # 필요한 key:
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "boolq": [
        {
            'instruction': '{paragraph}\n윗글로 미루어볼 때 다음 문장은 참인가 거짓인가?\n{question}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n가 주어졌을 때\n{question}\n을 판단해주세요. \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{paragraph}"는 "{question}"을 함의한다. 진위 여부를 판별하면? \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n{paragraph}\n안에는\n{question}\n라는 내용이 들어가 있다. 참 또는 거짓으로 대답해.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<보기>\n{paragraph}\n다음 문장은 참인가 거짓인가?\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '지문:\n{paragraph}\n질문:\n{question}\n{options}\n정답:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n"{question}"는 참인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '질문에 답하되, 지문에 근거하여 판단하세요.\n{question}은 옳은가?\n{paragraph}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 글\n{paragraph}\n을 보고 생각했을 때,\n{question}\n은 참이니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '참, 거짓 여부 판별\n\n근거: {paragraph}\n주장 또는 질문: {question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # 필요한 key:
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "sentineg": [
        {
            'instruction': '다음 문장의 감정을 긍정 또는 부정으로 분류해 줘.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n주어진 리뷰는 긍정적인가, 부정적인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n위 리뷰를 보고 감정을 분석하면?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 리뷰를 긍정, 부정으로 분석해봐.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 감성 분석하는 로봇입니다. 아래 문장의 분석 결과는?\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '화자의 감정 상태를 파악하세요.\n발화: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 쇼핑몰 후기의 감성을 분석해 줘.\n후기: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n이 문장이 상품에 대해 어떻게 생각하고 있는 것 같니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '쇼핑몰 후기 아무거나 한 문장 생성',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '{answer}적인 상품 리뷰를 짧게 하나 써 줘.',
            'input': '',
            'output': '{sentence}'
        },
    ],
    # 필요한 key:
    # word, eun_or_neun, context_1, context_2, options, answer
    # options = "선택지\n - 같은 뜻입니다.\n - 다른 뜻입니다."
    # answer = {다른 뜻입니다., 같은 뜻입니다.}
    "wic": [
        {
            'instruction': '"{context_1}"\n{context_2}\n두 문장에서 {word}{eun_or_neun} 같은 뜻인가, 다른 뜻인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 두 문맥에서 {word}{eun_or_neun} 같은 뜻으로 쓰였는지 알려주세요.\n"{context_1}"\n"{context_2}"\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': "'{context_1}' 그리고 '{context_2}'에서 {word}는 동일한 뜻으로 사용되었는지 판단하면?\n{options}",
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '1: {context_1}\n2: {context_2}\n1과 2에서 {word}{eun_or_neun} 같은 뜻으로 쓰였어?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context_1}"\n{context_2}\n두 문장에서 쓰인 {word}{eun_or_neun} 같은 뜻으로 쓰였나요, 아니면 다른 뜻으로 쓰였나요?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 문맥\n(1) {context_1}\n(2) {context_2}\n에서 {word}{eun_or_neun} 같은 뜻이니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '단어 {word}가 다음 두 문맥에서 같은 뜻으로 쓰였는지 구분해 봐.\n1. {context_1}\n2. {context_2}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{word}는 같은 뜻으로 쓰였습니까?\n문장 1: {context_1}\n문장 2: {context_2}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문맥 두 개가 주어진다. 단어 {word}가 같은 뜻으로 쓰였는지 판단하시오. \n{context_1}\n{context_2}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{word}{eun_or_neun} 여러 뜻을 가진다.\n문장 1: {context_1}\n문장 2: {context_2}\n\n문장 1과 2에서 {word}{eun_or_neun} 같은 뜻이에요?\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ]
}


def _make_options_str(*options):
    """utility function that make {options} form easily. Returns raw string."""
    l = ['선택지:']
    for option in options:
        l.append(f' - {option}')
    return '\n'.join(l)


def _process_copa(template, **raw_data):
    """returns raw string, which contains raw data and processed extra data, by template-formed."""
    data = copy.deepcopy(raw_data)
    label = data['label']
    question = data['question'].strip()
    data['options'] = _make_options_str(data['alternative_1'], data['alternative_2'])
    data['answer'] = data[f'alternative_{label + 1}']  # since label 0 means alternative_1
    if question == '원인':
        data['euro_or_ro'] = '으로'
        data['eun_or_neun'] = '은'
    elif question == '결과':
        data['euro_or_ro'] = '로'
        data['eun_or_neun'] = '는'
    else:
        raise NotImplementedError(f"unexpected raw data question: '{question}'")
    return {k: v.format_map(data) for k, v in template.items()}


def _process_hellaswag(template, **raw_data):
    """returns raw string, which contains raw data and processed extra data, by template-formed."""
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str(
        raw_data['ending_1'],
        raw_data['ending_2'],
        raw_data['ending_3'],
        raw_data['ending_4'],
    )
    data['answer'] = raw_data[f'ending_{label + 1}']
    return {k: v.format_map(data) for k, v in template.items()}


def _process_boolq(template, **raw_data):
    """returns raw string, which contains raw data and processed extra data, by template-formed."""
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str('거짓', '참')
    data['answer'] = ['거짓', '참'][label]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_sentineg(template, **raw_data):
    """returns raw string, which contains raw data and processed extra data, by template-formed."""
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str('부정', '긍정')
    data['answer'] = ['부정', '긍정'][label]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_wic(template, **raw_data):
    """returns raw string, which contains raw data and processed extra data, by template-formed."""
    data = copy.deepcopy(raw_data)
    label = data['label']
    if label == 0:
        answer = '다른 뜻입니다.'
    elif label == 1:
        answer = '같은 뜻입니다.'
    else:
        raise NotImplementedError
    data['answer'] = answer
    data['options'] = _make_options_str('다른 뜻입니다.', '같은 뜻입니다.')
    data['eun_or_neun'] = bojosa(data['word'])
    return {k: v.format_map(data) for k, v in template.items()}

