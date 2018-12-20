---
layout: post
title: "Digit Sum Recursion"
excerpt: "not what you are thinkin"
tags:
  - Streams
---

## Hackerrank Recursion Problem

[Problem Link](https://www.hackerrank.com/challenges/recursive-digit-sum/problem)

Here is the solution in java 8

```java

// Complete the superDigit function below.
    static int superDigit(String n, int k) {
        n = n.chars().mapToLong(Character::getNumericValue).sum()*k+"";
        if(n.length() == 1) return Integer.valueOf(n);
        else return superDigit(n,1);

    }

```

