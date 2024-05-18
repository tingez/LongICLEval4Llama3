import os
import re
import json
import typer
import requests
from tqdm import tqdm
from decouple import config
from typing import List, Optional
from collections import defaultdict
from datasets import load_dataset
from pydantic import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from sklearn.metrics import confusion_matrix

from enum import Enum
from pydantic import BaseModel, ValidationError

from llm import TensorRTLlama3
from utils import selectData

app = typer.Typer(pretty_exceptions_show_locals=False)


class IntentEnum(str, Enum):
        activate_my_card = "activate_my_card"
        age_limit = "age_limit"
        apple_pay_or_google_pay = "apple_pay_or_google_pay"
        atm_support = "atm_support"
        automatic_top_up = "automatic_top_up"
        balance_not_updated_after_bank_transfer = "balance_not_updated_after_bank_transfer"
        balance_not_updated_after_cheque_or_cash_deposit = "balance_not_updated_after_cheque_or_cash_deposit"
        beneficiary_not_allowed = "beneficiary_not_allowed"
        cancel_transfer = "cancel_transfer"
        card_about_to_expire = "card_about_to_expire"
        card_acceptance = "card_acceptance"
        card_arrival = "card_arrival"
        card_delivery_estimate = "card_delivery_estimate"
        card_linking = "card_linking"
        card_not_working = "card_not_working"
        card_payment_fee_charged = "card_payment_fee_charged"
        card_payment_not_recognised = "card_payment_not_recognised"
        card_payment_wrong_exchange_rate = "card_payment_wrong_exchange_rate"
        card_swallowed = "card_swallowed"
        cash_withdrawal_charge = "cash_withdrawal_charge"
        cash_withdrawal_not_recognised = "cash_withdrawal_not_recognised",
        change_pin = "change_pin"
        compromised_card = "compromised_card"
        contactless_not_working = "contactless_not_working"
        country_support = "country_support"
        declined_card_payment = "declined_card_payment"
        declined_cash_withdrawal = "declined_cash_withdrawal"
        declined_transfer = "declined_transfer"
        direct_debit_payment_not_recognised = "direct_debit_payment_not_recognised"
        disposable_card_limits = "disposable_card_limits"
        edit_personal_details = "edit_personal_details"
        exchange_charge = "exchange_charge"
        exchange_rate = "exchange_rate"
        exchange_via_app = "exchange_via_app"
        extra_charge_on_statement = "extra_charge_on_statement"
        failed_transfer = "failed_transfer"
        fiat_currency_support = "fiat_currency_support"
        get_disposable_virtual_card = "get_disposable_virtual_card"
        get_physical_card = "get_physical_card"
        getting_spare_card = "getting_spare_card"
        getting_virtual_card = "getting_virtual_card"
        lost_or_stolen_card = "lost_or_stolen_card"
        lost_or_stolen_phone = "lost_or_stolen_phone"
        order_physical_card = "order_physical_card"
        passcode_forgotten = "passcode_forgotten"
        pending_card_payment = "pending_card_payment"
        pending_cash_withdrawal = "pending_cash_withdrawal"
        pending_top_up = "pending_top_up"
        pending_transfer = "pending_transfer"
        pin_blocked = "pin_blocked"
        receiving_money = "receiving_money"
        Refund_not_showing_up = "Refund_not_showing_up"
        request_refund = "request_refund"
        reverted_card_payment = "reverted_card_payment"
        supported_cards_and_currencies = "supported_cards_and_currencies"
        terminate_account = "terminate_account"
        top_up_by_bank_transfer_charge = "top_up_by_bank_transfer_charge"
        top_up_by_card_charge = "top_up_by_card_charge"
        top_up_by_cash_or_cheque = "top_up_by_cash_or_cheque"
        top_up_failed = "top_up_failed"
        top_up_limits = "top_up_limits"
        top_up_reverted = "top_up_reverted"
        topping_up_by_card = "topping_up_by_card"
        transaction_charged_twice = "transaction_charged_twice"
        transfer_fee_charged = "transfer_fee_charged"
        transfer_into_account = "transfer_into_account"
        transfer_not_received_by_recipient = "transfer_not_received_by_recipient"
        transfer_timing = "transfer_timing"
        unable_to_verify_identity = "unable_to_verify_identity"
        verify_my_identity = "verify_my_identity",
        verify_source_of_funds = "verify_source_of_funds"
        verify_top_up = "verify_top_up"
        virtual_card_not_working = "virtual_card_not_working"
        visa_or_mastercard = "visa_or_mastercard"
        why_verify_identity = "why_verify_identity"
        wrong_amount_of_cash_received = "wrong_amount_of_cash_received"
        wrong_exchange_rate_for_cash_withdrawal = "wrong_exchange_rate_for_cash_withdrawal"




all_labels = [
        "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
        "change_pin", "compromised_card", "contactless_not_working", "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
        "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
        "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
        "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
        "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund", "reverted_card_payment", "supported_cards_and_currencies", "terminate_account",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
        "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
        "verify_source_of_funds", "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
        ]

class IntentModel(BaseModel):
    intent: IntentEnum


def __composeSystemPrompt(demo_data):
    """
        for Llama3-70B-Instruct, it follow the following instruction format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {{ model_answer_1 }}<|eot_id|>
    """
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                + "{system_prompt}\n\n{format_instructions}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                + "{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt = PromptTemplate.from_template(template=prompt_template)

    system_prompt = 'Yor are executing a intent extraction task. ' \
                + 'Given a customer service query, please predict the intent of the query. ' \
                + 'The predict answer must come from the demonstration examples without any extra words. ' \
                +' DO NOT add extra sentence and word. '
    system_prompt += 'You can only make prediction from the following categories: '
    system_prompt += ', '.join(all_labels) +'.\n'

    if len(demo_data) > 0:
        system_prompt += 'The examples are as follows:\n'

    for data in demo_data:
        system_prompt  += "service query: " + data['text'] + '\n```\n' + json.dumps({'intent': all_labels[data['label']]}) + '\n```\n'

    partial_prompt = prompt.partial(system_prompt=system_prompt)
    return partial_prompt


@app.command()
def exec_eval(shot_round:int=1):
    y_pred = []
    y_true = []
    SERVICE_HOST = config('SERVICE_HOST')
    SERVICE_PORT = config('SERVICE_PORT')
    DEBUG = config('DEBUG', default=False, cast=bool)

    assert SERVICE_HOST and SERVICE_PORT, 'please config SERVICE_HOST and SERVICE_HOST to the tensorrt server'

    dataset = load_dataset('banking77')

    train_data = list(dataset['train'])
    test_data = list(dataset['test'])


    demo_data = selectData(train_data, 'label', shot_round)
    eval_data = selectData(test_data, 'label', 2)

    parser = PydanticOutputParser(pydantic_object=IntentModel)

    system_prompt = __composeSystemPrompt(demo_data)
    prompt = system_prompt.partial(format_instructions=parser.get_format_instructions())

    llama3_70b = TensorRTLlama3(host=SERVICE_HOST, port=SERVICE_PORT)

    for item in tqdm(eval_data):
        y_true.append(all_labels[item['label']])

        curr_prompt = 'service query: ' + item['text'] + '\nintent: '
        prompt_str = prompt.format(user_msg=curr_prompt)

        response = llama3_70b.invoke(prompt_str)
        try:
            result = parser.parse(response)
            y_pred.append(result.intent.value)
        except OutputParserException as e:
            y_pred.append('Other')

        if DEBUG:
            print(f'{item["text"]}: true:{y_true[-1]}, predict:{y_pred[-1]}')

    total_labels = all_labels[:]
    total_labels.append('Other')
    rst = confusion_matrix(y_true, y_pred, labels=total_labels)
    assert len(y_true) == len(y_pred), 'length of ground true does not match with length of predication'
    print(rst)

    acc = 0
    for idx in range(len(y_true)):
        if y_true[idx] == y_pred[idx]:
            acc += 1
    print(f'acc : {acc/len(y_true):.4f}')


if __name__ == '__main__':
    app()
