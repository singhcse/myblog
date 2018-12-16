---
layout: post
title: "Mapper in Stream"
excerpt: "Streams | map()"
tags:
  - Lambda Expressions
  - Streams
last_modified_at: 2017-12-10T13:48:50+05:30
---

## Stream Function mapper

mapper is a stateless function which is applied to each element and the function returns the new stream.

In this example we will see how mapper can convert one Object to another Object.
In simple words we can create objects using flowing data.

Here we are creating PubgPlayers object by passing names in constructor of class using map() method.

Look at the code its HALWA.


```java

package com.tachyon.basic;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class StreamMapper {

	static class PubgPlayers{
		String country = "India";
		String name;
		public PubgPlayers(String name) {
			this.name = name;
		}
		public String getCountry() {
			return country;
		}
		public void setCountry(String country) {
			this.country = country;
		}
		public String getName() {
			return name;
		}
		public void setName(String name) {
			this.name = name;
		}
		@Override
		public String toString() {
			return "PubgPlayers [country=" + country + ", name=" + name + "]";
		}
		
		
	}
	
	public static void main(String[] args) {
		List<String> names = Arrays.asList("Shroud","Mortal","Dynamo","kronten");
			
		List<PubgPlayers> indianPlayer = names.stream()
				.filter(i -> !i.startsWith("S"))
				// make new PubgPlayers object using names
				.map(i -> {
					PubgPlayers player = new PubgPlayers(i);
					return player;
				})
				// storing as a list of players
				.collect(Collectors.toList());
		indianPlayer.forEach(System.out::println);
	}
	
}

```

We can make it more shorter using this 

```java
List<PubgPlayers> indianPlayer = names.stream()
				.filter(i -> !i.startsWith("S"))
				// make new PubgPlayers object using names
				.map(i -> {
					return new PubgPlayers(i);
				})
				// storing as a list of players
				.collect(Collectors.toList());

```

Even more shorten using this 

```java
List<PubgPlayers> indianPlayer = names.stream()
				.filter(i -> !i.startsWith("S"))
				// make new PubgPlayers object using names
				.map(PubgPlayers::new)
				// storing as a list of players
				.collect(Collectors.toList());

```

will add more content
