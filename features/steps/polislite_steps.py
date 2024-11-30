from behave import given, when, then
import numpy as np
from polislite import analyze_opinions


@given("the following statements and votes")
def step_impl(context):
    context.statements = []
    vote_map = {"agree": 1, "disagree": -1}
    votes = []

    for row in context.table:
        if row["statements"] != "statements":  # Skip header row
            context.statements.append(row["statements"])
            vote_row = [vote_map[row[user]] for user in row.headings[1:]]
            votes.append(vote_row)

    context.votes = np.array(votes).T  # Transpose to match expected format


@when("I analyze the opinions")
def step_impl(context):
    context.results = analyze_opinions(context.statements, context.votes)


@then('the statement "{statement}" should show strong consensus')
def step_impl(context, statement):
    consensus_data = dict(
        zip(
            context.statements,
            [(score, agree) for _, score, agree in context.results["consensus_data"]],
        )
    )
    score, agreement_level = consensus_data[statement]
    assert score > 0.8, f"{statement} should have high agreement"
    assert agreement_level < 0.3, f"{statement} should have low variance"


@then('the statement "{statement}" should be more divisive')
def step_impl(context, statement):
    divisive_data = dict(
        zip(
            context.statements, [agree for _, agree in context.results["divisive_data"]]
        )
    )
    assert divisive_data[statement] > 0.3, f"{statement} should show higher variance"


@then("there should be exactly {count:d} opinion groups identified")
def step_impl(context, count):
    actual_groups = len(context.results["group_data"])
    assert actual_groups == count, f"Expected {count} groups but found {actual_groups}"
