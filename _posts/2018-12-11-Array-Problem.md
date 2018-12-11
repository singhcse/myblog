---
layout: post
title: "Rain Water Tap Problem"
categories:
  - "Programming"
not_main: true
hidden: true
last_modified_at: 2018-12-11T20:31:52+05:30
---

This is an array related problem in which we have to calculate the maximum unit of water can be stored.here array elements are representing the height of building blocks.

input : array
output : maximum unit of water
```java

package com.tachyon.basic;

public class ArrayProblems {
	public static int maxUnit(int[] buildings) {
		int count = 0;
		int n = buildings.length;
		int[] left = new int[n];
		int[] right = new int[n];
		left[0] =buildings[0];right[n-1]=buildings[n-1];
		for(int i = 1;i<n-1;i++) {
			left[i]=Math.max(left[i-1], buildings[i]);
		}
		for(int j = n-2;j>=0;j--) {
			right[j]=Math.max(right[j+1], buildings[j]);
		}
		
		for(int k=0;k<n-1;k++) {
			count+=Math.min(left[k],right[k])-buildings[k];
		}
		
		return count;
	}
	public static void main(String []args){
		System.out.println(maxUnit(new int[]{1,0,2,1,3}));
	}

}

```

output : 3

