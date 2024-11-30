Feature: Opinion Analysis
  Scenario: Basic opinion analysis with sample data
    Given the following statements and votes:
        | statements  | user1  | user2  | user3  | user4   | user5   | user6   |
        | Statement A | agree  | agree  | agree  | agree   | agree   | agree   |
        | Statement B | agree  | agree  | agree  | disagree| disagree| disagree|
    When I analyze the opinions
    Then the statement "Statement A" should show strong consensus
    And the statement "Statement B" should be more divisive
    And there should be exactly 2 opinion groups identified
