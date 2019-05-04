package org.github.jelmerk.foo;

public interface SearchResult<ITEM extends Item> {

    float getDistance();

    ITEM getItem();
}
