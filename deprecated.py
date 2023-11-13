from random import choice
from korean_utils import bojosa


def make_boolq_example(paragraph, question, label):
    assert isinstance(label, int)
    validity = '참' if label == 1 else '거짓'
    returns = [
        {
            'instruction': f"{paragraph}\n이 지문에 따르면\n{question}\n은 맞는 말입니까?",
            'input': '',
            'output': f'지문에 따르면, "{question}"은 {validity}입니다.'
        },
        {
            'instruction': f"{paragraph}\n이 지문에 따르면\n{question}\n에 대해 참/거짓으로 대답해봐.",
            'input': '',
            'output': f'"{question}"은/는 {validity}입니다.'
        },
        {
            'instruction': f"{question}은\n{paragraph}\n이 지문에 따르면 참이니 거짓이니?",
            'input': '',
            'output': f'{validity}입니다.'
        },
        {
            'instruction': f"{paragraph}\n여기서, \n{question}\n의 질문에 대답해봐.",
            'input': '',
            'output': f'지문에 따르면, {validity}입니다.'
        },
        {
            'instruction': f"주어진 지문에 근거하여 질문에 대해 참/거짓 여부를 답하시오.\n지문:\n{paragraph}\n질문\n{question}",
            'input': '',
            'output': f'{validity}'
        },
    ]
    return choice(returns)


def make_copa_example(premise, question, alternative_1, alternative_2, label):
    assert isinstance(label, int)
    question = question.strip()
    # try:
    assert question in ['원인', '결과']
    # except:
    #     print(question)
    #     raise
    q = "원인으" if question == '원인' else question
    jx = '은' if question == '원인' else '는'
    arr = [alternative_1, alternative_2]
    gana = ["가", "나"]
    returns = [
        {
            'instruction': f"다음 상황이 주어졌을 때, 이 상황의 {q}로 적절한 것을 고르시오.\n상황: {premise}\n\n 1. {alternative_1}\n  2. {alternative_2}\n",
            'input': '',
            'output': f'상황 "{premise}"의 {question}{jx} {arr[label]}입니다.'
        },
        {
            'instruction': f"'{premise}'\n\n위의 현상에 대하여 {q}로 알맞은 선택지는?\n  1. {alternative_1}\n  2. {alternative_2}\n",
            'input': '',
            'output': f'상황 "{premise}"의 {question}{jx} {arr[label]}입니다.'
        },
        {
            'instruction': f"생각해 봐! '{premise}'\n\n이 때, {q}{jx} 뭘까? 1, 2번 중에서 골라봐!!\n  1. {alternative_1}\n  2. {alternative_2}\n",
            'input': '',
            'output': f'{label + 1}. {arr[label]}'
        },
        {
            'instruction': f"'{premise}'에 대한 알맞은 {q}{jx}?\n  가) {alternative_1}\n  나) {alternative_2}\n",
            'input': '',
            'output': f'옳은 {q}{jx}\n{gana[label]}) {arr[label]}\n입니다.'
        },
        {
            'instruction': f"'{premise}'에 대한 알맞은 {q}{jx}?\n단, 번호로만 대답하세요.\n  0. {alternative_1}\n  1. {alternative_2}\n",
            'input': '',
            'output': f'{label}'
        },
    ]
    return choice(returns)


def make_hellaswag_example(context, ending_1, ending_2, ending_3, ending_4, label):
    assert label in [0, 1, 2, 3]
    endings = [ending_1, ending_2, ending_3, ending_4]
    returns = [
        {
            'instruction': f'''다음의 글을 읽고 물음에 답하세요.\n\n\n{context}\n글에서 이어질 문장으로 가장 올바른 것은?\n
1. {ending_1}
2. {ending_2}
3. {ending_3}
4. {ending_4}
''',
            'input': '',
            'output': f'주어진 글에 따르면, 이어질 문장으로 가장 올바른 것은 {label + 1}. "{endings[label]}"입니다.'
        },
        {
            'instruction': f'''"{context}"\n이어질 말로 제일 적당한 문장을 골라 줘.\n
0. {ending_1}
1. {ending_2}
2. {ending_3}
3. {ending_4}''',
            'input': '',
            'output': f'이어질 말로 가장 올바른 문장은 다음과 같습니다.\n{label}. "{endings[label]}"'
        },
        {
            'instruction': f'''"{context}" 에 가장 어울리는 다음 말은?\n\n
1. {ending_1}
2. {ending_2}
3. {ending_3}
4. {ending_4}''',
            'input': '',
            'output': f'"{endings[label]}"입니다.'
        },
        {
            'instruction': f'''
1. {ending_1}
2. {ending_2}
3. {ending_3}
4. {ending_4}
여기서 문장을 골라서 다음 글을 완성하세요.
{context}
''',
            'input': '',
            'output': f'{context} {endings[label]}'
        },
    ]
    return choice(returns)


def make_sentineg_example(sentence, label):
    assert label in [0, 1]
    word = ['부정', '긍정']
    returns = [
        {
            'instruction': f'이 문장의 감정 상태를 부정, 긍정으로 분류하세요.\n{sentence}',
            'input': '',
            'output': f'{word[label]}입니다.'
        },
        {
            'instruction': f'"{sentence}"\n이 문장을 말한 사람의 감정 상태를 긍정 또는 부정으로 분류하세요.',
            'input': '',
            'output': f'"{sentence}"의 감정은 {word[label]}입니다.'
        },
        {
            'instruction': f'"{sentence}"\n이 문장은 부정적인 문장일까요, 긍정적인 문장일까요?',
            'input': '',
            'output': f'"{sentence}"는 {word[label]}적인 문장입니다.'
        },
        {
            'instruction': f'주어지는 발화를 보고 감정 상태를 분류해 봐.\n"{sentence}"\n',
            'input': '',
            'output': f'주어진 문장 "{sentence}"은 "{word[label]}"입니다.'
        },
        {
            'instruction': f'이 말을 한 사람의 감정 상태는? \n{sentence}',
            'input': '',
            'output': f'"{word[label]}적인 감정을 갖고 있을 가능성이 높습니다.'
        },
    ]
    return choice(returns)


def make_wic_example(word, context_1, context_2, label):
    assert label in [0, 1]
    same = ['다른 의미', '같은 의미']
    same2 = ['다른 뜻', '같은 뜻']
    ne = ['아니오', '네']
    ne2 = ['아니요', '그렇습니다']
    ne3 = ['아니요', '네']
    ne3 = ['아니요', '맞습니다']
    returns = [
        {
            'instruction': f'{context_1}\n{context_2}\n\n위 두 문장에서 쓰인 {word}는 같은 의미입니까? 아니면 다른 의미입니까?',
            'input': '',
            'output': f'{word}{bojosa(word)}\n제시된 두 문장에서 {same[label]}로 사용되었습니다.'
        },
        {
            'instruction': f'{context_1}\n{context_2}\n\n위 두 문맥을 보고 판단할 때, {word}{bojosa(word)} 같은 의미인가요?',
            'input': '',
            'output': f'{ne}. 두 문맥에서 {same[label]}로 사용되었습니다.'
        },
        {
            'instruction': f'{context_1}\n{context_2}\n\n여기서 {word}{bojosa(word)} 동일한 뜻인지 가르쳐 주세요~!',
            'input': '',
            'output': f'두 문장에서 {word}{bojosa(word)} {same2[label]}입니다.'
        },
        {
            'instruction': f'여기서 {word}는 똑같은 뜻인지 가르쳐 주세요~!\n{context_1}\n{context_2}',
            'input': '',
            'output': f'제시하신 두 문장에서 {word}{bojosa(word)} 각자 {same[label]}입니다.'
        },
        {
            'instruction': f'{context_1}\n{context_2}\n여기서 {word}{bojosa(word)} 같은 뜻인지 알려줄 수 있어?',
            'input': '',
            'output': f'물론입니다! 두 문장에서 {word}{bojosa(word)} {same2[label]}으로 사용되었습니다.'
        },
    ]
    return choice(returns)
